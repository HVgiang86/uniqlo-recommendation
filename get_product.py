import requests
import csv

# Define the API URL (replace this with the actual API endpoint)
api_url = 'https://uniqlo-be-f348.onrender.com/v2/products'

# Fetch the data from the API
response = requests.get(api_url)
data = response.json()

# Open a CSV file for writing
with open('products.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(['ID', 'Name', 'Price', 'Description', 'Specifications', 'Rating', 'Discount', 'Brand Name', 'Colors'])

    # Check if data contains the product list
    if 'data' in data:
        products = data['data']

        # Iterate through each product
        for product in products:
            product_id = product.get('id', '')
            name = product.get('name', '')
            price = product.get('price', '')
            description = product.get('description', '')
            specifications = product.get('specifications', '')
            rating = product.get('averageRating', '')
            discount = product.get('discountPercentage', '')
            brand_name = product.get('brand', {}).get('name', '')

            # Extract colors from variations
            variations = product.get('variations', [])
            colors = ', '.join([variation.get('color', '') for variation in variations])

            # Write the row to the CSV file
            writer.writerow([product_id, name, price, description, specifications, rating, discount, brand_name, colors])

print("Products have been written to 'products.csv'")
