import json
from collections import Counter
import spacy
import matplotlib.pyplot as plt
import os

def analyze_semantic_imbalance(ann_file, sample_size=None):
    print(f"Loading annotations from {ann_file}...")
    with open(ann_file) as f:
        data = json.load(f)
    
    captions = [a['caption'] for a in data['annotations']]
    
    if sample_size:
        captions = captions[:sample_size]
        print(f"Analyzing a subset of {sample_size} captions for speed...")
    else:
        print(f"Analyzing all {len(captions)} captions...")

    # Load English NLP model for Part-of-Speech tagging
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Error: spaCy model not found. Please run: python -m spacy download en_core_web_sm")
        return

    noun_counter = Counter()
    
    # Common hedge phrases in VizWiz that indicate low quality/unanswerability
    hedge_phrases = [
        "too blurry", "blurry", "cannot see", "can't see", 
        "unanswerable", "dark", "hard to tell", "difficult to see",
        "out of focus", "not visible", "quality issues are too severe to recognize visual content"
    ]
    hedge_counter = Counter()

    print("Extracting nouns and analyzing linguistic bias (this may take a minute)...")
    
    # Process captions through spaCy pipeline
    # nlp.pipe is much faster than calling nlp() in a loop
    for doc in nlp.pipe((cap.lower() for cap in captions), batch_size=1000):
        # 1. Extract Objects (Nouns)
        nouns = [token.lemma_ for token in doc if token.pos_ == "NOUN" and len(token.text) > 1]
        noun_counter.update(nouns)
        
        # 2. Check for Hedge Phrases
        text = doc.text
        for phrase in hedge_phrases:
            if phrase in text:
                hedge_counter.update([phrase])

    total_nouns = sum(noun_counter.values())
    total_captions = len(captions)

    # --- Print Terminal Report ---
    print("\n" + "="*50)
    print("VIZWIZ SEMANTIC IMBALANCE REPORT")
    print("="*50)
    
    print("\nTHE 'HEAD' (Top 100 Most Frequent Objects):")
    print("These are over-represented. The model will likely over-predict these.")
    for word, freq in noun_counter.most_common(2500):
        print(f"   - {word}: {freq} ({freq / total_nouns * 100:.2f}%)")
        with open('./txt_files/head_objects.txt', 'a') as f:
            f.write(f"{word}\n")

    print("\nTHE 'LONG TAIL' (Sample of Rare Objects):")
    print("These appear rarely. The model will struggle to learn their visual features.")
    # Get items that appear exactly 2 times (avoiding typos that appear once)
    rare_items = [word for word, freq in noun_counter.items() if freq == 2]
    for word in rare_items:
        print(f"   - {word}")
        with open('./txt_files/rare_objects_sample.txt', 'a') as f:
            f.write(f"{word}\n")

    print("\nLINGUISTIC BIAS (Hedge Phrases):")
    print("Frequency of captions explicitly mentioning image degradation:")
    for phrase, freq in hedge_counter.most_common():
        print(f"   - '{phrase}': {freq} captions ({freq / total_captions * 100:.2f}%)")

    # --- Plot the Long Tail Distribution ---
    plot_long_tail(noun_counter)

def plot_long_tail(noun_counter):
    """Generates a visualization of the class imbalance."""
    # Sort frequencies descending
    frequencies = sorted(noun_counter.values(), reverse=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, color='red', linewidth=2)
    plt.fill_between(range(len(frequencies)), frequencies, color='red', alpha=0.3)
    
    plt.title("VizWiz Object (Noun) Frequency Distribution", fontsize=14)
    plt.xlabel("Object Rank (from most common to least common)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.yscale('log') # Log scale is crucial to actually see the long tail
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # Annotate head vs tail
    plt.axvline(x=100, color='black', linestyle='--')
    plt.text(110, max(frequencies)*0.5, '← The "Head" (Common)\nThe "Long Tail" (Rare) →', fontsize=11)
    
    output_file = 'vizwiz_long_tail_plot.png'
    plt.savefig(output_file)
    print(f"\nSaved distribution plot to '{output_file}'")

if __name__ == '__main__':
    import sys
    default_ann = '../datasets/vizwiz/annotations/train.json'
    ann = sys.argv[1] if len(sys.argv) > 1 else default_ann
    
    if os.path.exists(ann):
        analyze_semantic_imbalance(ann)
    else:
        print(f"Could not find annotation file at {ann}")