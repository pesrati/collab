# Импорт необходимых библиотек
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from textblob import TextBlob

# Инициализация токенизатора и модели
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id


# Функция для генерации ответа
def generate_response(input_text, character):
    inputs = tokenizer(
        character + ": " + input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    attention_mask = inputs["attention_mask"]
    inputs = inputs["input_ids"]

    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        do_sample=True,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Обрезаем ответ после первого предложения
    response = response.split("\n")[0]

    return response


# Функция для определения настроения
def determine_sentiment(input_text):
    blob = TextBlob(input_text)
    if blob.sentiment.polarity > 0.1:
        return "positive"
    else:
        return "negative"


# Тестирование
def test():
    test_cases = test_cases = [
        "Hello, how are you body?",
        "I don't like this.",
        "This is wonderful!",
        "I don't like this.",
        "I am happy.",
        "I am sad.",
        "This is delightful!",
        "This is terrible.",
        "I love this.",
        "I hate this.",
    ]
    for test_case in test_cases:
        sentiment = determine_sentiment(test_case)
        if sentiment == "positive":
            response = generate_response(test_case, "Batman")
        else:
            response = generate_response(test_case, "Joker")
        print(f"Вопрос: {test_case}\nОтвет: {response}\n")


test()
