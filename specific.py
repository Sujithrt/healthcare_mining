import asyncio
from urllib.parse import urljoin
import json
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

async def extract_a_tag_text(page):
    soup = BeautifulSoup(await page.content(), 'html.parser')
    disease_info_list = []

    ul_tag = soup.find('ul', class_='link-list')

    if ul_tag:
        a_tags = ul_tag.find_all('a')

        for a_tag in a_tags:
            disease_info = {}
            disease_info['name'] = a_tag.get_text().strip()

            href = a_tag.get('href')
            if href:
                linked_url = urljoin(page.url, href)
                disease_info['link'] = linked_url
                linked_page = await page.goto(linked_url)
                linked_page_content = await linked_page.text()

                linked_soup = BeautifulSoup(linked_page_content, 'html.parser')
                article_page_h2_tags = linked_soup.select('.article-page h2')

                symptoms = []
                prevention = []
                causes = []

                for h2_tag in article_page_h2_tags:
                    h2_text = h2_tag.get_text().lower()

                    if 'symptoms' in h2_text:
                        symptoms_elements = h2_tag.find_next('ul').find_all('li')
                        symptoms = [symptom_element.get_text().strip() for symptom_element in symptoms_elements]

                    elif any(keyword in h2_text for keyword in ['prevent', 'prevention', 'treat', 'treatment']):
                        prevention_elements = h2_tag.find_next('ul').find_all('li')
                        prevention = [prevention_element.get_text().strip() for prevention_element in prevention_elements]

                    elif any(keyword in h2_text for keyword in ['cause', 'causes']):
                        causes_elements = h2_tag.find_next('ul').find_all('li')
                        causes = [cause_element.get_text().strip() for cause_element in causes_elements]

                # Modify the disease_info based on array comparisons
                if symptoms == causes:
                    causes = []  # Clear causes if same as symptoms
                elif prevention == causes:
                    causes = []  # Clear causes if same as prevention
                elif symptoms == prevention:
                    prevention = []  # Clear prevention if same as symptoms

                # Assign values to disease_info if they are not empty
                if symptoms:
                    disease_info['symptoms'] = symptoms
                if prevention:
                    disease_info['prevention'] = prevention
                if causes:
                    disease_info['causes'] = causes

                # Ensure at least one category remains before adding to the list
                if 'symptoms' in disease_info or 'prevention' in disease_info or 'causes' in disease_info:
                    disease_info_list.append(disease_info)

    return disease_info_list


async def scrape_playwright(url, letter):
    print(f"Started scraping for letter '{letter}'...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url.format(letter))

            disease_info_list = await extract_a_tag_text(page)

            # Save the data to a JSON file
            with open(f'{letter}.json', 'w', encoding='utf-8') as json_file:
                json.dump(disease_info_list, json_file, ensure_ascii=False, indent=4)

            print(f"Data saved to '{letter}.json'")
        except Exception as e:
            print(f"Error during scraping for letter '{letter}': {e}")
        finally:
            await browser.close()

async def scrape_multiple_pages(url, letters):
    for letter in letters:
        await scrape_playwright(url, letter)

# Specify the letters you want to scrape
letters_to_scrape = ['g','z']

# Modify the URL to include the placeholder for letters
if __name__ == "__main__":
    base_url = "https://www.webmd.com/a-to-z-guides/health-topics?pg={}"
    asyncio.run(scrape_multiple_pages(base_url, letters_to_scrape))
