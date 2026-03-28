#!/usr/bin/env python3
"""
Create Diverse Test Set - General knowledge questions in Esperanto

Creates QA test questions based on factual content found in the corpus.
Questions cover: American presidents, scientists, inventors, writers, sports.
All questions and answers are in Esperanto.
"""

import json

# Questions based on factual sentences found in corpus
QUESTIONS = [
    # American Presidents
    {
        "id": 1,
        "question": "Kiu estis Lincoln?",
        "answer": "usona prezidento",
        "expected_keywords": ["prezidento", "usona"],
        "question_type": "WHO",
        "difficulty": "easy",
        "category": "american_history",
        "notes": "Corpus: 'Lincoln estas konsiderata, kun Washington, unu el la du plej gravaj usonaj prezidentoj'"
    },
    {
        "id": 2,
        "question": "Kiu estis Thomas Jefferson?",
        "answer": "tria prezidento de Usono",
        "expected_keywords": ["prezidento", "tria", "usono"],
        "question_type": "WHO",
        "difficulty": "medium",
        "category": "american_history",
        "notes": "Corpus: 'Thomas Jefferson (1743-1826) - la tria prezidento de Usono'"
    },
    {
        "id": 3,
        "question": "Kiu estis Roosevelt?",
        "answer": "usona prezidento",
        "expected_keywords": ["prezidento", "usona"],
        "question_type": "WHO",
        "difficulty": "easy",
        "category": "american_history",
        "notes": "Corpus: multiple mentions of Roosevelt as president"
    },
    {
        "id": 4,
        "question": "Kiuj estis la plej gravaj usonaj prezidentoj?",
        "answer": "Lincoln kaj Washington",
        "expected_keywords": ["lincoln", "washington", "prezidento"],
        "question_type": "WHO",
        "difficulty": "medium",
        "category": "american_history",
        "notes": "Corpus: 'Lincoln estas konsiderata, kun Washington, unu el la du plej gravaj usonaj prezidentoj'"
    },

    # Scientists
    {
        "id": 5,
        "question": "Kiu formulis la ĉefajn leĝojn de la dinamiko?",
        "answer": "Newton",
        "expected_keywords": ["newton", "leĝoj", "dinamiko"],
        "question_type": "WHO",
        "difficulty": "medium",
        "category": "science",
        "notes": "Corpus: 'Newton kiu formulis la ĉefajn leĝojn de la dinamiko'"
    },
    {
        "id": 6,
        "question": "Kiujn Nobel-premiojn ricevis Marie Curie?",
        "answer": "fiziko kaj kemio",
        "expected_keywords": ["fiziko", "kemio", "nobel"],
        "question_type": "WHAT",
        "difficulty": "hard",
        "category": "science",
        "notes": "Corpus: 'Marie Curie estas ĝis nun la nura persono kiu ricevis Nobel-premiojn en du diferencaj sciencoj (fiziko en 1903, kemio en 1911)'"
    },
    {
        "id": 7,
        "question": "Kiu malkovris radioaktivecon?",
        "answer": "Marie Curie",
        "expected_keywords": ["curie", "marie", "radioaktiv"],
        "question_type": "WHO",
        "difficulty": "medium",
        "category": "science",
        "notes": "Corpus: 'Maria Skłodowska-Curie... malkovrintino de radioaktivaj elementoj'"
    },
    {
        "id": 8,
        "question": "Kiu kreis la teorion de evolucio?",
        "answer": "Darwin",
        "expected_keywords": ["darwin", "evolucio", "teorio"],
        "question_type": "WHO",
        "difficulty": "medium",
        "category": "science",
        "notes": "Corpus: 'la teorio de Darwin tuj disvastiĝis rapide tra la scienca komunumo'"
    },
    {
        "id": 9,
        "question": "Kio estas natura selektado?",
        "answer": "proceso per kiu specioj evoluas",
        "expected_keywords": ["darwin", "evolucio", "specio"],
        "question_type": "WHAT",
        "difficulty": "hard",
        "category": "science",
        "notes": "Corpus: 'Darwin teoriigis ke la specioj floras aŭ mortas kiam estas subigitaj al procezoj de natura selektado'"
    },
    {
        "id": 10,
        "question": "Kion studis Galileo?",
        "answer": "rapidon, graviton, movadon",
        "expected_keywords": ["rapido", "gravito", "movo"],
        "question_type": "WHAT",
        "difficulty": "medium",
        "category": "science",
        "notes": "Corpus: 'Galileo studis rapidon kaj rapidecon, graviton kaj la fenomenon de libera falo'"
    },

    # Inventors
    {
        "id": 11,
        "question": "Kiu inventis la telefonon?",
        "answer": "Bell",
        "expected_keywords": ["bell", "telefono", "inventis"],
        "question_type": "WHO",
        "difficulty": "easy",
        "category": "technology",
        "notes": "Corpus: 'La unua tre primitiva telefono estis inventita... Bell... en 1876 Bell pliefikigis la telefonon'"
    },
    {
        "id": 12,
        "question": "Kiu estis Thomas Edison?",
        "answer": "usona inventisto",
        "expected_keywords": ["edison", "inventisto", "usona"],
        "question_type": "WHO",
        "difficulty": "easy",
        "category": "technology",
        "notes": "Corpus: 'Thomas Alva Edison, usona inventisto'"
    },
    {
        "id": 13,
        "question": "Kiu estis Nikola Tesla?",
        "answer": "serba-usona inĝeniero kaj inventisto",
        "expected_keywords": ["tesla", "inventisto", "inĝeniero"],
        "question_type": "WHO",
        "difficulty": "medium",
        "category": "technology",
        "notes": "Corpus: 'Nikola Tesla, serba-usona inĝeniero kaj inventisto, aŭtoro de 300 patentoj'"
    },
    {
        "id": 14,
        "question": "Kiom da patentoj havis Tesla?",
        "answer": "300",
        "expected_keywords": ["300", "patentoj", "tesla"],
        "question_type": "HOW_MANY",
        "difficulty": "hard",
        "category": "technology",
        "notes": "Corpus: 'Nikola Tesla... aŭtoro de 300 patentoj'"
    },

    # Writers
    {
        "id": 15,
        "question": "Kiu verkis Don Kiĥoton?",
        "answer": "Cervantes",
        "expected_keywords": ["cervantes", "kiĥot", "hispana"],
        "question_type": "WHO",
        "difficulty": "medium",
        "category": "literature",
        "notes": "Corpus mentions Cervantes as important Spanish writer (Don Quixote context)"
    },

    # Sports
    {
        "id": 16,
        "question": "Kiuj sportoj estas usonaj inventoj?",
        "answer": "basketbalo kaj flugpilko",
        "expected_keywords": ["basketbalo", "flugpilko", "usona"],
        "question_type": "WHAT",
        "difficulty": "hard",
        "category": "sports",
        "notes": "Corpus: 'basketbalo, flugpilko... estas usonaj inventoj'"
    },
    {
        "id": 17,
        "question": "Kio estas basketbalo?",
        "answer": "usona invento, sporto",
        "expected_keywords": ["sporto", "usona", "invento"],
        "question_type": "WHAT",
        "difficulty": "easy",
        "category": "sports",
        "notes": "Corpus: 'basketbalo... estas usonaj inventoj'"
    },

    # General Knowledge (from diverse topics)
    {
        "id": 18,
        "question": "Kiu studis la teorion de gravito?",
        "answer": "Newton",
        "expected_keywords": ["newton", "gravito", "teorio"],
        "question_type": "WHO",
        "difficulty": "easy",
        "category": "science",
        "notes": "Corpus: 'Edmond Halley uzis la teorion de gravito, antaŭnelonge malkovrita de Isaac Newton'"
    },
    {
        "id": 19,
        "question": "En kiu jarcento vivis Newton?",
        "answer": "17-a jarcento",
        "expected_keywords": ["17", "jarcento", "newton"],
        "question_type": "WHEN",
        "difficulty": "medium",
        "category": "science",
        "notes": "Corpus: 'En la 17-a jarcento okazis forta antaŭeniro de la Fiziko danke al la rezultoj atingitaj de Newton'"
    },
    {
        "id": 20,
        "question": "Kiu ricevis Nobel-premion en du sciencoj?",
        "answer": "Marie Curie",
        "expected_keywords": ["curie", "marie", "nobel", "du"],
        "question_type": "WHO",
        "difficulty": "medium",
        "category": "science",
        "notes": "Corpus: 'Marie Curie estas ĝis nun la nura persono kiu ricevis Nobel-premiojn en du diferencaj sciencoj'"
    },

    # Additional diverse questions
    {
        "id": 21,
        "question": "Kiu murdis Kennedy?",
        "answer": "Jack Ruby",
        "expected_keywords": ["ruby", "jack", "murdis", "kennedy"],
        "question_type": "WHO",
        "difficulty": "hard",
        "category": "american_history",
        "notes": "Corpus: 'Jack Ruby, usona murdisto ligita kun atenco kontraŭ John Fitzgerald Kennedy'"
    },
    {
        "id": 22,
        "question": "Kio estas la Vendo de Luiziano?",
        "answer": "vendo de tero de Francio al Usono",
        "expected_keywords": ["luiziano", "vendo", "francoj", "usonanoj"],
        "question_type": "WHAT",
        "difficulty": "hard",
        "category": "american_history",
        "notes": "Corpus: 'La Vendo de Luiziano fare de la francoj al la usonanoj sub la Prezidento Thomas Jefferson en 1803'"
    },
    {
        "id": 23,
        "question": "Kiam okazis la Vendo de Luiziano?",
        "answer": "1803",
        "expected_keywords": ["1803", "luiziano"],
        "question_type": "WHEN",
        "difficulty": "medium",
        "category": "american_history",
        "notes": "Corpus: 'La Vendo de Luiziano... en 1803'"
    },
    {
        "id": 24,
        "question": "Kiu estis prezidento dum la Vendo de Luiziano?",
        "answer": "Thomas Jefferson",
        "expected_keywords": ["jefferson", "thomas", "prezidento"],
        "question_type": "WHO",
        "difficulty": "medium",
        "category": "american_history",
        "notes": "Corpus: 'La Vendo de Luiziano... sub la Prezidento Thomas Jefferson'"
    },
    {
        "id": 25,
        "question": "Kion faris Bell?",
        "answer": "inventis la telefonon",
        "expected_keywords": ["bell", "telefono", "inventis"],
        "question_type": "WHAT",
        "difficulty": "easy",
        "category": "technology",
        "notes": "Corpus: 'Bell... en 1876 Bell pliefikigis la telefonon kaj starigis telefonsistemon'"
    },
    {
        "id": 26,
        "question": "Kiam Bell inventis la telefonon?",
        "answer": "1876",
        "expected_keywords": ["1876", "bell", "telefono"],
        "question_type": "WHEN",
        "difficulty": "hard",
        "category": "technology",
        "notes": "Corpus: 'en 1876 Bell pliefikigis la telefonon'"
    },
    {
        "id": 27,
        "question": "Kio estas futbalo?",
        "answer": "sporto, ludo per piedpilko",
        "expected_keywords": ["sporto", "ludo", "pilko"],
        "question_type": "WHAT",
        "difficulty": "easy",
        "category": "sports",
        "notes": "Corpus: 'piedpilko = ludo per piedpilko, futbalo'"
    },
    {
        "id": 28,
        "question": "Kiun lingvon instruas Instituto Cervantes?",
        "answer": "hispanan lingvon",
        "expected_keywords": ["hispana", "lingvo", "cervantes"],
        "question_type": "WHAT",
        "difficulty": "medium",
        "category": "language",
        "notes": "Corpus: 'Instituto Cervantes estas komisiita de hispana registaro instrui la lingvon tutmonde'"
    },
    {
        "id": 29,
        "question": "Kiu estis John F. Kennedy?",
        "answer": "usona prezidento",
        "expected_keywords": ["prezidento", "usona", "kennedy"],
        "question_type": "WHO",
        "difficulty": "easy",
        "category": "american_history",
        "notes": "Corpus: 'John Fitzgerald Kennedy' mentioned as US president"
    },
    {
        "id": 30,
        "question": "Kiuj estis Winston Churchill kaj Roosevelt?",
        "answer": "politikaj gvidantoj dum mondmilito",
        "expected_keywords": ["churchill", "roosevelt", "prezidento", "ĉefministro"],
        "question_type": "WHO",
        "difficulty": "medium",
        "category": "world_history",
        "notes": "Corpus: 'Winston Churchill kaj Prezidento Roosevelt' mentioned together"
    }
]

def main():
    output_path = 'data/test_sets/qa_test_diverse_30.jsonl'

    print(f"Creating diverse test set: {len(QUESTIONS)} questions")
    print(f"Output: {output_path}\n")

    # Count by category
    by_category = {}
    for q in QUESTIONS:
        cat = q['category']
        by_category[cat] = by_category.get(cat, 0) + 1

    print("Questions by category:")
    for cat, count in sorted(by_category.items()):
        print(f"  {cat}: {count}")

    print()

    # Write JSONL
    with open(output_path, 'w') as f:
        for q in QUESTIONS:
            # Remove notes field for actual test set
            test_q = {k: v for k, v in q.items() if k != 'notes'}
            f.write(json.dumps(test_q, ensure_ascii=False) + '\n')

    print(f"✓ Created {output_path}")
    print(f"  {len(QUESTIONS)} questions covering:")
    print(f"    - American history: {by_category.get('american_history', 0)}")
    print(f"    - Science: {by_category.get('science', 0)}")
    print(f"    - Technology: {by_category.get('technology', 0)}")
    print(f"    - Sports: {by_category.get('sports', 0)}")
    print(f"    - Literature: {by_category.get('literature', 0)}")
    print(f"    - World history: {by_category.get('world_history', 0)}")
    print(f"    - Language: {by_category.get('language', 0)}")

if __name__ == '__main__':
    main()
