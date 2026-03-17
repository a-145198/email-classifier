import google.generativeai as genai
genai.configure(api_key="AIzaSyAQBRbMeKiyT18PLbc151QcbajZ4obu4xQ")
model = genai.GenerativeModel("models/gemini-pro-latest")
response = model.generate_content("Say hello")
print(response.text)
