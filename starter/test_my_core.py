import json

from traffic_core import forward, viterbi, accuracy_against_ground_truth


# Đọc config
with open("data/hmm_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)


# Một chuỗi observation mẫu
observations = ["LOW", "LOW", "MEDIUM", "HIGH", "HIGH"]


# Chạy Forward
result = forward(observations, config)


# In kết quả
for t, probabilities in enumerate(result):
    print(f"t = {t}")
    print(probabilities)
    print("sum =", sum(probabilities.values()))
    print()

path, probability = viterbi(observations, config)

print("Viterbi path:")
print(path)

print("Probability:")
print(probability)
print()


print("Accuracy")
estimated = [
    "LOW",
    "LOW",
    "MEDIUM",
    "HIGH",
    "HIGH"
]

true = [
    "LOW",
    "LOW",
    "MEDIUM",
    "MEDIUM",
    "HIGH"
]

accuracy = accuracy_against_ground_truth(estimated, true)

print("Accuracy:", accuracy)
print("Accuracy %:", accuracy * 100)