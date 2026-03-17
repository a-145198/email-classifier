import google.generativeai as genai
model = genai.GenerativeModel("models/gemini-pro-latest")
response = model.generate_content("Say hello")
print(response.text)
