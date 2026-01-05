# Retriever Comparison Tool

## Overview

`compare_retrievers.py` is a comprehensive benchmarking tool that compares all available slot-based retrievers on the same queries, measuring:

- **Speed**: Query latency (min/avg/max)
- **Memory**: Peak memory usage
- **CPU**: Peak CPU usage
- **Quality**: Result overlap analysis

## Features

✅ **Progress bars** - Real-time progress with tqdm  
✅ **Resource tracking** - Memory and CPU monitoring  
✅ **Selective testing** - Choose specific retrievers  
✅ **Custom queries** - Use your own test queries  
✅ **JSON export** - Save detailed results

## Usage

### Compare All Available Retrievers (Default)

```bash
python scripts/compare_retrievers.py --index data/indexes/slot_full
```

### Compare Specific Retrievers

```bash
# Test only HNSW and ScaNN
python scripts/compare_retrievers.py --index data/indexes/slot_full --retrievers hnsw,scann

# Test only fast retrievers
python scripts/compare_retrievers.py --index data/indexes/slot_full --retrievers hybrid,hnsw,scann
```

### Adjust Number of Results

```bash
# Get top 20 results instead of default 10
python scripts/compare_retrievers.py --index data/indexes/slot_full -k 20
```

### Use Custom Queries

```bash
# Create queries file (one per line, format: "eo_query | en_translation")
echo "Kiu inventis la telefonon? | Who invented the telephone?" > my_queries.txt
echo "Kie estas Parizo? | Where is Paris?" >> my_queries.txt

python scripts/compare_retrievers.py --index data/indexes/slot_full --queries my_queries.txt
```

### Save Results to JSON

```bash
python scripts/compare_retrievers.py \
  --index data/indexes/slot_full \
  --output benchmark_results/comparison_$(date +%Y%m%d_%H%M%S).json
```

## Available Retrievers

| Name | Key | Requirements |
|------|-----|--------------|
| MemoryMapped | `mmap` | `mmap/` directory |
| MultiFAISS | `multifaiss` | `multifaiss/` directory |
| Hybrid | `hybrid` | `faiss/` + `mmap/` |
| HNSW | `hnsw` | `hnsw/` + `mmap/` |
| ScaNN | `scann` | `scann/` + `mmap/` |

The script auto-detects which retrievers are available based on index files.

## Example Output

```
Testing retrievers: hnsw, scann

HNSW        : 100%|█████████████████| 4/4 [00:01<00:00,  2.85query/s, last=53.6ms, mem=3022MB]
ScaNN       : 100%|█████████████████| 4/4 [00:00<00:00,  4.19query/s, last=14.7ms, mem=2104MB]

====================================================================================================
RETRIEVER COMPARISON RESULTS
====================================================================================================

Retriever       Avg Time     Min        Max        Memory       CPU      Queries 
----------------------------------------------------------------------------------------------------
ScaNN                 37.3ms     14.7ms     68.6ms     2103.6MB    0.0%       4
HNSW                 149.8ms     53.6ms    299.2ms     3022.4MB    0.0%       4

Rankings:
  🥇 Fastest:       ScaNN (37.3ms avg)
  💾 Lowest Memory: ScaNN (2103.6MB peak)
```

## Latest Benchmark Results (4.2M docs)

| Retriever | Avg Latency | Memory Peak | Notes |
|-----------|-------------|-------------|-------|
| **ScaNN** | **18.0ms** ⭐ | 2.1GB | Fastest, highest accuracy (90-95% expected) |
| **HNSW** | 65.7ms | 3.0GB | Fast, simple, good accuracy (85-90%) |
| **Hybrid** | 82.2ms | **1.1GB** 💾 | FAISS + mmap, good accuracy (90%) |
| **MultiFAISS** | 101.0ms | 1.3GB | 3 separate FAISS indexes (75%) |
| MemoryMapped | 5021ms | 4.6GB | Slowest (no pre-filtering) |

## Tips

- **First run** takes longer due to indexer loading (offset building)
- **Progress bars** update every query with timing and memory
- **Garbage collection** runs between retrievers to free memory
- **Resource tracking** shows peak values (not continuous monitoring)

## Notes

- All retrievers use the same embedding models and slot extraction
- Results may vary based on:
  - Query complexity
  - Index parameters (ef_search, num_leaves_to_search, etc.)
  - System load
  - Available RAM
