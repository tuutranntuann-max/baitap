import requests

# URL of a sample image from a more reliable source
image_url = "https://picsum.photos/id/237/200/300" # A simple placeholder image
image_filename = "sample_image.jpg"

# Download the image
response = requests.get(image_url)
if response.status_code == 200:
    with open(image_filename, "wb") as f:
        f.write(response.content)
    print(f"Downloaded {image_filename} successfully.")
else:
    print(f"Failed to download image from {image_url}. Status code: {response.status_code}")
