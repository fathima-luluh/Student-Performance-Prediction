import matplotlib.pyplot as plt

plt.scatter(df["study_hours"], df["previous_marks"], c=df["passed"])
plt.xlabel("Study Hours")
plt.ylabel("Previous Marks")
plt.title("Student Performance")
plt.show()