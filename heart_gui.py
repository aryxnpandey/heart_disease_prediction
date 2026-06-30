
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

try:
    df = pd.read_csv("heart.csv")
except FileNotFoundError:
    messagebox.showerror("Error","heart.csv not found.\nPlace heart.csv in the same folder as this file.")
    raise SystemExit

features = df.drop("target", axis=1)
target = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))*100

app = tk.Tk()
app.title("Heart Disease Prediction System")
app.geometry("900x760")
app.configure(bg="#EAF6FF")
app.resizable(False, False)

tk.Label(app,text="❤ HEART DISEASE PREDICTION SYSTEM",
         bg="#0B5394",fg="white",
         font=("Arial",22,"bold"),
         pady=12).pack(fill="x")

tk.Label(app,text="Artificial Intelligence & Machine Learning",
         bg="#EAF6FF",
         fg="#0B5394",
         font=("Arial",12)).pack(pady=5)

tk.Label(app,text=f"Model Accuracy : {accuracy:.2f}%",
         bg="#EAF6FF",
         fg="green",
         font=("Arial",11,"bold")).pack()

frame=tk.LabelFrame(app,text="Patient Information",
                    bg="#EAF6FF",
                    font=("Arial",12,"bold"),
                    padx=15,pady=15)
frame.pack(fill="x",padx=20,pady=15)

field_names=[
("Age","age"),("Blood Pressure","bp"),("Cholesterol","chol"),
("Fasting Blood Sugar (0/1)","fbs"),("Maximum Heart Rate","thalach"),
("Old Peak","oldpeak"),("CA","ca")
]
entries={}
for r,(lab,key) in enumerate(field_names):
    tk.Label(frame,text=lab,bg="#EAF6FF").grid(row=r,column=0,sticky="w",padx=5,pady=5)
    e=tk.Entry(frame,width=18)
    e.grid(row=r,column=1,padx=5,pady=5)
    entries[key]=e

gender=tk.IntVar(value=1)
tk.Label(frame,text="Gender",bg="#EAF6FF").grid(row=0,column=2,sticky="w")
tk.Radiobutton(frame,text="Male",variable=gender,value=1,bg="#EAF6FF").grid(row=0,column=3)
tk.Radiobutton(frame,text="Female",variable=gender,value=0,bg="#EAF6FF").grid(row=0,column=4)

cp_map={"Typical Angina":0,"Atypical Angina":1,"Non-anginal":2,"Asymptomatic":3}
ecg_map={"Normal":0,"ST-T Abnormal":1,"LV Hypertrophy":2}
slope_map={"Upsloping":0,"Flat":1,"Downsloping":2}
thal_map={"Normal":1,"Fixed Defect":2,"Reversible Defect":3}

def make_combo(text,row,values):
    tk.Label(frame,text=text,bg="#EAF6FF").grid(row=row,column=2,sticky="w")
    c=ttk.Combobox(frame,values=list(values.keys()),state="readonly",width=18)
    c.current(0)
    c.grid(row=row,column=3,columnspan=2,pady=5)
    return c

cp=make_combo("Chest Pain",1,cp_map)
ecg=make_combo("Rest ECG",2,ecg_map)
slope=make_combo("Slope",3,slope_map)
thal=make_combo("Thal",4,thal_map)

exang=tk.IntVar(value=0)
tk.Label(frame,text="Exercise Angina",bg="#EAF6FF").grid(row=5,column=2,sticky="w")
tk.Radiobutton(frame,text="No",variable=exang,value=0,bg="#EAF6FF").grid(row=5,column=3)
tk.Radiobutton(frame,text="Yes",variable=exang,value=1,bg="#EAF6FF").grid(row=5,column=4)

result=tk.Label(app,text="Waiting for Prediction...",bg="#EAF6FF",
                fg="blue",font=("Arial",15,"bold"))
result.pack(pady=15)

risk=tk.Label(app,text="",bg="#EAF6FF",font=("Arial",12))
risk.pack()

def clear():
    for e in entries.values():
        e.delete(0,tk.END)
    result.config(text="Waiting for Prediction...",fg="blue")
    risk.config(text="")

def predict():
    try:
        patient=pd.DataFrame([[
            int(entries["age"].get()),
            gender.get(),
            cp_map[cp.get()],
            int(entries["bp"].get()),
            int(entries["chol"].get()),
            int(entries["fbs"].get()),
            ecg_map[ecg.get()],
            int(entries["thalach"].get()),
            exang.get(),
            float(entries["oldpeak"].get()),
            slope_map[slope.get()],
            int(entries["ca"].get()),
            thal_map[thal.get()]
        ]],columns=features.columns)

        pred=model.predict(patient)[0]
        prob=model.predict_proba(patient)[0][1]*100

        if prob<30:
            cat="Low Risk"
        elif prob<70:
            cat="Moderate Risk"
        else:
            cat="High Risk"

        if pred==1:
            result.config(text="❤ Heart Disease Detected",fg="red")
        else:
            result.config(text="✔ No Heart Disease Detected",fg="green")

        risk.config(text=f"Risk: {prob:.2f}%   |   {cat}")

    except ValueError:
        messagebox.showerror("Input Error","Please enter valid numeric values.")

btn=tk.Frame(app,bg="#EAF6FF")
btn.pack(pady=20)

tk.Button(btn,text="Predict",command=predict,bg="#28A745",fg="white",width=12).grid(row=0,column=0,padx=5)
tk.Button(btn,text="Clear",command=clear,bg="#FFC107",width=12).grid(row=0,column=1,padx=5)
tk.Button(btn,text="Exit",command=app.destroy,bg="#DC3545",fg="white",width=12).grid(row=0,column=2,padx=5)

tk.Label(app,text="Developed by Aryan Pandey",
         bg="#EAF6FF",fg="gray").pack(side="bottom",pady=10)

app.mainloop()
