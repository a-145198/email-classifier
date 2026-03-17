import pandas as pd
import requests
import time

# ======= CONFIGURE =======
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
]  # Try each in order if desired

INPUT_CSV = "/Users/adeshnarayanatellakua/Documents/mini_project/datasets/enron_fully_processed.csv"
OUTPUT_CSV = "enron_labeled_llm_max500.csv"

categories = [
    "Action Required", "Informational/Updates", "Social/Personal", "Scheduling/Events",
    "Support/Help", "Promotional/Marketing", "Transactional/Confirmations",
    "Urgent/Time-Sensitive", "Spam/Junk"
]
MAX_PER_CATEGORY = 500

def label_with_groq(subject, body, model_name):
    prompt = (
        f"Classify this email into one of these categories: {', '.join(categories)}.\n"
        "Only reply with the category name.\n"
        f"Subject: {subject}\nBody: {body}"
    )
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        label = resp.json()["choices"][0]["message"]["content"].strip().split('\n')[0]
        if label in categories:
            return label
    except Exception as e:
        print(f"[Groq-{model_name} error] {e}")
    return None

def auto_label_max500(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df = df.sample(frac=1, random_state=42)  # Shuffle for fairness!
    results = []
    category_count = {cat: 0 for cat in categories}

    pd.DataFrame([], columns=["subject", "body", "llm_label", "llm_label_source", "llm_label_time"]).to_csv(output_csv, index=False)

    for i, row in df.iterrows():
        subject = str(row.get("subject", ""))
        body = str(row.get("body", ""))
        label, source = None, None

        # Try Groq models in order
        for groq_model in GROQ_MODELS:
            label = label_with_groq(subject, body, groq_model)
            if label is not None:
                source = groq_model
                break

        if label is None:
            label = "Unlabeled"
            source = "None"

        if label in categories and category_count[label] < MAX_PER_CATEGORY:
            results.append({
                "subject": subject,
                "body": body,
                "llm_label": label,
                "llm_label_source": source,
                "llm_label_time": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            category_count[label] += 1

        if all([v >= MAX_PER_CATEGORY for v in category_count.values()]):
            print("All category limits reached. Stopping.")
            break

        if len(results) % 25 == 0:
            pd.DataFrame(results).to_csv(output_csv, index=False)
            print(f"{len(results)} labeled/saved, progress written.")

        time.sleep(0.5)

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Final output saved to {output_csv}. Category totals:", category_count)

if __name__ == "__main__":
    auto_label_max500(INPUT_CSV, OUTPUT_CSV)