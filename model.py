from datasets import load_dataset
dataset = load_dataset("MakTek/Customer_support_faqs_dataset")

# Assuming your dataset is in the variable `dataset`

# Initialize empty lists for questions and answers
questions = []
answers = []

# Loop through each item in the dataset
for item in dataset['train']:
    questions.append(item['question'])
    answers.append(item['answer'])

# Create the new data structure
data = {
    'questions': questions,
    'answers': answers
}

# Now `data` contains the reformatted dataset
print(data)


from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import torch

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def convert_to_text_dataset(dataset, tokenizer, file_path="train.txt"):
    with open(file_path, "w") as f:
        for question, answer in zip(dataset["questions"], dataset["answers"]):
            f.write(f"Question: {question}\nAnswer: {answer}\n\n")
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )

# Prepare the training dataset
train_dataset = convert_to_text_dataset(data, tokenizer)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Fine-tune the model
print('training started')
trainer.train()

# Save the model
model.save_pretrained("fine-tuned-gpt2")
tokenizer.save_pretrained("fine-tuned-gpt2")


def generate_response(question):
    input_text = f"Question: {question}\nAnswer:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer.split("Answer:")[1].strip()

# Example usage
question = "What is your return policy?"
answer = generate_response(question)
print(answer)