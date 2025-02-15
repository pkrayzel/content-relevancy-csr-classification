import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
import trafilatura
from transformers import pipeline
import pandas as pd


def fetch_and_save_html(url, output_file):
    """Fetch webpage content using a simple HTTP request."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(response.text)
            print(f"Plain HTML content saved to {output_file}")
        else:
            print(f"Failed to fetch {url}, status code: {response.status_code}")
    except Exception as e:
        print(f"Error fetching {url}: {e}")


def render_and_save_html(url, output_file):
    """Render webpage using Selenium and save content."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(url)

        # Attempt to click any 'Accept Cookies' button if present
        try:
            cookie_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept')]"))
            )
            cookie_button.click()
            print("Clicked 'Accept Cookies' button.")
        except Exception as e:
            print(f"No cookie button found or clicked: {e}")

        time.sleep(5)

        html_content = driver.page_source
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(html_content)

        print(f"Rendered HTML content saved to {output_file}")

    finally:
        driver.quit()


def extract_sentences(html_content):
    """Extract sentences from HTML content."""
    text = trafilatura.extract(html_content)
    sentences = text.split('. ') if text else []
    return sentences


def classify_sentences(sentences, title, url, classifier):
    """Classify sentences as relevant or not."""
    results = []
    for sentence in sentences:
        context = f"Title: {title} URL: {url} Sentence: {sentence}"
        result = classifier(context)
        results.append((sentence, result[0]['label'], result[0]['score']))
    return results


def process_urls(urls, output_dir="website_outputs", model_path="fine_tuned_distilbert"):
    """Process multiple URLs by fetching, rendering, and classifying content."""
    os.makedirs(output_dir, exist_ok=True)

    classifier = pipeline("text-classification", model=model_path)

    results = []

    for url in urls:
        print(f"Processing {url}")

        # Prepare file paths
        safe_url = url.replace("https://", "").replace("http://", "").replace("/", "_")
        plain_file = os.path.join(output_dir, f"{safe_url}_plain.html")
        rendered_file = os.path.join(output_dir, f"{safe_url}_rendered.html")

        # Fetch plain HTML
        fetch_and_save_html(url, plain_file)

        # Render with Selenium
        render_and_save_html(url, rendered_file)

        # Extract and classify sentences
        with open(plain_file, "r", encoding="utf-8") as f:
            plain_sentences = extract_sentences(f.read())
        with open(rendered_file, "r", encoding="utf-8") as f:
            rendered_sentences = extract_sentences(f.read())

        # Generate title for classification (could be improved with actual title extraction)
        title = "Webpage Analysis"

        # Classify sentences
        plain_results = classify_sentences(plain_sentences, title, url, classifier)
        rendered_results = classify_sentences(rendered_sentences, title, url, classifier)

        # Count relevant sentences
        plain_relevant = sum(1 for _, label, _ in plain_results if label == "LABEL_1")
        rendered_relevant = sum(1 for _, label, _ in rendered_results if label == "LABEL_1")

        # Append results to the summary
        results.append({
            "url": url,
            "plain_relevant": plain_relevant,
            "rendered_relevant": rendered_relevant,
            "difference": rendered_relevant - plain_relevant
        })

    # Save summary to CSV
    results_df = pd.DataFrame(results)
    csv_file = os.path.join(output_dir, "classification_summary.csv")
    results_df.to_csv(csv_file, index=False)
    print(f"Classification summary saved to {csv_file}")


# Example usage
urls_to_process = [
    "https://support.xbox.com/en-US/help/account-profile/profile/change-xbox-live-gamertag",
    "https://www.python.org/",
    "https://en.wikipedia.org/wiki/Web_scraping",
    "https://www.duckduckgo.com/"
]

process_urls(urls_to_process) 
