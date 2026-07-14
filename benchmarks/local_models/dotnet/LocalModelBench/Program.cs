using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;
using LLama;
using LLama.Common;

var options = Options.Parse(args);
var jsonOptions = new JsonSerializerOptions
{
    PropertyNameCaseInsensitive = true,
    PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
};
var modelConfig = JsonSerializer.Deserialize<ModelConfig>(File.ReadAllText(options.ModelsPath), jsonOptions)!;
var prompts = JsonSerializer.Deserialize<List<PromptSpec>>(File.ReadAllText(options.PromptsPath), jsonOptions)!;

Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(options.OutputPath)) ?? ".");
await using var output = new StreamWriter(options.OutputPath, false);

foreach (var model in modelConfig.Gguf)
{
    var modelPath = Environment.GetEnvironmentVariable(model.PathEnv);
    if (string.IsNullOrWhiteSpace(modelPath))
    {
        Console.Error.WriteLine($"Skipping {model.Id} because {model.PathEnv} is not set");
        continue;
    }

    var loadTimer = Stopwatch.StartNew();
    var parameters = new ModelParams(modelPath)
    {
        ContextSize = 4096,
        GpuLayerCount = 999
    };
    using var weights = LLamaWeights.LoadFromFile(parameters);
    loadTimer.Stop();

    foreach (var prompt in prompts)
    {
        using var context = weights.CreateContext(parameters);
        var executor = new InteractiveExecutor(context);
        var inferenceParams = new InferenceParams
        {
            MaxTokens = options.MaxTokens,
            AntiPrompts = []
        };

        var generated = new List<string>();
        var cpuBefore = Process.GetCurrentProcess().TotalProcessorTime;
        var generationTimer = Stopwatch.StartNew();
        await foreach (var token in executor.InferAsync(prompt.Prompt, inferenceParams))
        {
            generated.Add(token);
        }
        generationTimer.Stop();

        var process = Process.GetCurrentProcess();
        var peakRssBytes = process.PeakWorkingSet64 > 0 ? process.PeakWorkingSet64 : process.WorkingSet64;
        var row = new ResultRow(
            Runtime: "dotnet_llamasharp",
            ModelId: model.Id,
            PromptId: prompt.Id,
            Purpose: prompt.Purpose,
            ModelPath: modelPath,
            LoadSeconds: loadTimer.Elapsed.TotalSeconds,
            GenerationSeconds: generationTimer.Elapsed.TotalSeconds,
            CpuSeconds: (Process.GetCurrentProcess().TotalProcessorTime - cpuBefore).TotalSeconds,
            TokensGenerated: generated.Count,
            TokensPerSecond: generationTimer.Elapsed.TotalSeconds > 0
                ? generated.Count / generationTimer.Elapsed.TotalSeconds
                : 0,
            PeakRssMb: peakRssBytes / 1024.0 / 1024.0,
            Output: string.Concat(generated)
        );

        await output.WriteLineAsync(JsonSerializer.Serialize(row, jsonOptions));
        await output.FlushAsync();
    }
}

record Options(string ModelsPath, string PromptsPath, string OutputPath, int MaxTokens)
{
    public static Options Parse(string[] args)
    {
        string? models = null;
        string? prompts = null;
        string? output = null;
        var maxTokens = 160;

        for (var i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--models":
                    models = args[++i];
                    break;
                case "--prompts":
                    prompts = args[++i];
                    break;
                case "--output":
                    output = args[++i];
                    break;
                case "--max-tokens":
                    maxTokens = int.Parse(args[++i]);
                    break;
            }
        }

        if (models is null || prompts is null || output is null)
        {
            throw new ArgumentException("Usage: --models models.json --prompts prompts.json --output results.jsonl [--max-tokens 160]");
        }

        return new Options(models, prompts, output, maxTokens);
    }
}

record ModelConfig([property: JsonPropertyName("gguf")] List<GgufModel> Gguf);
record GgufModel(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("path_env")] string PathEnv);
record PromptSpec(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("purpose")] string Purpose,
    [property: JsonPropertyName("prompt")] string Prompt);

record ResultRow(
    string Runtime,
    string ModelId,
    string PromptId,
    string Purpose,
    string ModelPath,
    double LoadSeconds,
    double GenerationSeconds,
    double CpuSeconds,
    int TokensGenerated,
    double TokensPerSecond,
    double PeakRssMb,
    string Output);
