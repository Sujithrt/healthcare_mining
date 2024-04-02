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
            disease_info['name'] = a_tag.get_text()

            href = a_tag.get('href')
            if href:
                linked_url = urljoin(page.url, href)
                disease_info['link'] = page.url  # Assuming you intentionally record the page's URL, not the linked URL
                linked_page = await page.goto(linked_url)
                linked_page_content = await linked_page.text()

                linked_soup = BeautifulSoup(linked_page_content, 'html.parser')
                article_page_h2_tags = linked_soup.select('.article-page h2')

                # Initialize keys to ensure they exist for the check later
                disease_info['symptoms'] = []
                disease_info['prevention'] = []
                disease_info['causes'] = []

                for h2_tag in article_page_h2_tags:
                    h2_text = h2_tag.get_text().lower()

                    if 'symptoms' in h2_text:
                        symptoms_elements = h2_tag.find_next('ul').find_all('li')
                        symptoms_list = [symptom_element.get_text() for symptom_element in symptoms_elements]
                        disease_info['symptoms'] = symptoms_list

                    elif any(keyword in h2_text for keyword in ['prevent', 'prevention', 'treat', 'treatment']):
                        prevention_elements = h2_tag.find_next('ul').find_all('li')
                        prevention_list = [prevention_element.get_text() for prevention_element in prevention_elements]
                        disease_info['prevention'] = prevention_list

                    elif any(keyword in h2_text for keyword in ['cause', 'causes']):
                        causes_elements = h2_tag.find_next('ul').find_all('li')
                        causes_list = [cause_element.get_text() for cause_element in causes_elements]
                        disease_info['causes'] = causes_list

                # Check if any of the important fields are empty or non-existent, skip appending if so
                if not disease_info['symptoms'] or not disease_info['prevention'] or not disease_info['causes']:
                    continue  # Skip this disease_info object if any critical info is missing

                disease_info_list.append(disease_info)

    return disease_info_list


async def scrape_playwright(url, letter):
    print(f"Started scraping for letter '{letter}'...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url.format(letter))

            disease_info_list = await extract_a_tag_text(page)

            # Save the data to a JSON file
            with open(f'{letter}.json', 'w', encoding='utf-8') as json_file:
                json.dump(disease_info_list, json_file, ensure_ascii=False, indent=4)

            print(f"Data saved to '{letter}.json'")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

async def scrape_multiple_pages(url):
    for letter in range(ord('a'), ord('z')+1):
        letter = chr(letter)
        await scrape_playwright(url, letter)

# TESTING
if __name__ == "__main__":
    url = "https://www.webmd.com/a-to-z-guides/health-topics?pg={}"
    asyncio.run(scrape_multiple_pages(url))
