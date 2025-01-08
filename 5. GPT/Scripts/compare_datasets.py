# Compare the length of input children test with input children train by counting characters
with open('input_childSpeech_testSet.txt', 'r', encoding='utf-8') as f:
    test_text = f.read()

with open('input_childSpeech_trainingSet.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Count the number of characters in each text
train_length = len(text)
test_length = len(test_text)

print(f"Length of training set: {train_length} characters")
print(f"Length of test set: {test_length} characters")
proportion = test_length / train_length
print(f"The test set is {proportion:.2f} times the size of the training set.")
# Other proportion
proportion = train_length / test_length
print(f"The training set is {proportion:.2f} times the size of the test set.")

# Compare the length of input children test with input children train and input Shakespeare by counting characters
with open('input_shakespeare.txt', 'r', encoding='utf-8') as f:
    shakespeare_text = f.read()

# Count the number of characters in Shakespeare text
shakespeare_length = len(shakespeare_text)

print(f"Length of Shakespeare set: {shakespeare_length} characters")

proportion = shakespeare_length / train_length
print(f"The Shakespeare set is {proportion:.2f} times the size of the training set.")
# Other proportion
proportion = train_length / shakespeare_length
print(f"The training set is {proportion:.2f} times the size of the Shakespeare set.")

# Create a sorted list of unique characters in the training text
chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"Unique characters in training set: {chars}")
print(f"Vocabulary size of training set: {vocab_size} characters")

# Repeat for test set
chars_test = sorted(list(set(test_text)))
vocab_size_test = len(chars_test)

print(f"Unique characters in test set: {chars_test}")
print(f"Vocabulary size of test set: {vocab_size_test} characters")

# Repeat for Shakespeare set
chars_shakespeare = sorted(list(set(shakespeare_text)))
vocab_size_shakespeare = len(chars_shakespeare)

print(f"Unique characters in Shakespeare set: {chars_shakespeare}")
print(f"Vocabulary size of Shakespeare set: {vocab_size_shakespeare} characters")

# Count the number of characters in Shakespeare text that are not in the training text
unknown_chars = 0
for char in shakespeare_text:
    if char not in chars:
        unknown_chars += 1

unknown_proportion = unknown_chars / shakespeare_length
print(f"{unknown_chars} characters in Shakespeare set are not in the training set.")
print(f"{unknown_proportion:.2%} of characters in Shakespeare set are unknown in the training set.")

# Read the texts from the training and test sets
with open('input_childSpeech_testSet.txt', 'r', encoding='utf-8') as f:
    test_words = f.read().split()

with open('input_childSpeech_trainingSet.txt', 'r', encoding='utf-8') as f:
    train_words = f.read().split()

# Create sets of unique words
unique_test_words = set(test_words)
unique_train_words = set(train_words)

# Find words that are in the test set but not in the training set
unknown_words = unique_test_words - unique_train_words

# Calculate the number and percentage of unknown words
unknown_count = len(unknown_words)
test_total_words = len(test_words)
unknown_percentage = (unknown_count / test_total_words) * 100

print(f"Number of unique words in test set: {len(unique_test_words)}")
print(f"Number of unique words in train set: {len(unique_train_words)}")
print(f"Number of words in test set not in train set: {unknown_count}")
print(f"Percentage of unknown words in test set: {unknown_percentage:.2f}%")
