from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

temperature_list = [0.0, 0.5, 1.0, 1.5, 2.0]

### Low temperature (0 - 0.3) = predictable, focused
# "The sky is blue."
# 👉 Always picks the most likely word — very factual and predictable.

### Medium temperature (0.7 - 1.0) = creative but sensible
# 🌡️ Temperature = 0.7
# "The sky is glowing with orange and pink hues."
# 👉 Still makes sense, but more descriptive and creative.

### High temperature (>1.0) = wild, chaotic, sometimes incoherent
# 🔥 Temperature > 1.0
# "The sky is whispering secrets of forgotten stars."
# 👉 Now it's imaginative or poetic — unusual but fun.

for temperature in temperature_list:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature)
    response = llm.invoke("Immediately i need a  \n complete the above sentence use your imagination")
    print(f"Temperature: {temperature} Response: {response.content}")

### Prompts tested
# I am going to buy
# Today Weather is
# Immediately i need a

### Creativity becomes wild as temperature increases.

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=2.0, top_k=1000)

prompt_wo_option_for_imagination = """
Context: With the 2026 Tamil Nadu assembly elections looming, the political landscape is buzzing, and the Tamilaga Vettri Kazhagam (TVK), led by actor Vijay, is poised to make a significant statement with its second state-level mega conference in Madurai. Following the success of their first convention in Villupuram, the TVK is pulling out all stops to ensure this Madurai event surpasses expectations and solidifies its presence in the state's political arena.
The decision to hold the conference in Madurai comes on the heels of crucial announcements from the TVK's working committee meeting held in Panayur, Chennai, on July 4th.
Question: will tvk make this conference big?
You are a helpful AI assistant that answers questions using the given context."""

prompt_wi_option_for_imagination = """
Context: With the 2026 Tamil Nadu assembly elections looming, the political landscape is buzzing, and the Tamilaga Vettri Kazhagam (TVK), led by actor Vijay, is poised to make a significant statement with its second state-level mega conference in Madurai. Following the success of their first convention in Villupuram, the TVK is pulling out all stops to ensure this Madurai event surpasses expectations and solidifies its presence in the state's political arena.
The decision to hold the conference in Madurai comes on the heels of crucial announcements from the TVK's working committee meeting held in Panayur, Chennai, on July 4th.
Question: whats your opinion about tvk?
Answer based on the context, but feel free to share your own interpretation."""

response = llm.invoke(prompt_wi_option_for_imagination)
print(response.content)

# https://rumn.medium.com/setting-top-k-top-p-and-temperature-in-llms-3da3a8f74832
# https://medium.com/@kelseyywang/a-comprehensive-guide-to-llm-temperature-%EF%B8%8F-363a40bbc91f

# ✅ Temperature changes the shape of the probability distribution (flatter or sharper).
# ✅ Top-K limits the number of tokens from the vocabulary when picking the next token. For example, 5 means the top 5 tokens (based on probability values) will be considered.
# ✅ Temperature and Top-K can be used together — temperature changes the probability distribution, and Top-K restricts the number of token choices. Together, they control whether the model picks more common (likely) or less common (creative) words.
# ✅ Top-p filters out tokens whose cumulative probability is less than a specified threshold