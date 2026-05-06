model = joblib.load("models/model.pkl")

new_student = [[5, 75, 60]]

prediction = model.predict(new_student)

print("Pass" if prediction[0] == 1 else "Fail")