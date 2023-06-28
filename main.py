import joblib
import pandas as pd
from functools import partial

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox
import tkinter.font as tkFont

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, confusion_matrix, auc, roc_curve

def train(x_train, y_train, x_test, y_test):
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    return clf

def preprocess(xtrain):
    scale = StandardScaler()
    xtrain = scale.fit_transform(xtrain)
    return xtrain

class MainWindow:
    def __init__(self, root):
        root.title("Prediction Model v1.2")
        width=600
        height=370
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        self.model=None
        self.trainloc, self.scanloc, self.mod=tk.StringVar(), tk.StringVar(), tk.IntVar()

        #Main Heading
        ft = tkFont.Font(family='Times',size=22)
        self.label1=tk.Label(root, font=ft,justify = LEFT, text = "Kidney Cancer Detection")
        self.label1.place(x=135,y=10)

        #choose train or model
        self.rbtrain = Radiobutton(root, text='Use pretrained Model', variable=self.mod, value=0, cursor="hand2", command=partial(self.trainvisible, False))
        self.rbtrain.place(x=20, y=80)
        self.rbpretrain = Radiobutton(root, text='Train a model', variable=self.mod, value=1, cursor="hand2",command=partial(self.trainvisible, True))
        self.rbpretrain.place(x=210, y=80)

        #Get Model Path
        ft = tkFont.Font(family='Times',size=10)
        self.label2=tk.Label(root,font = ft, justify = LEFT, text = "Path : ")
        self.label2.place(x=30,y=125)

        self.entrytraining=tk.Entry(root, font = ft, justify = LEFT, textvariable = self.trainloc)
        self.entrytraining.place(x=150,y=125,width=350,height=30)

        self.labelinvalidtrain = tk.Label(root,font = tkFont.Font(family='Times',size=9), justify = LEFT)
        self.labelinvalidtrain.place(x=150, y=155)

        buttontrainbrowse=tk.Button(root,font = ft, text = "Browse", cursor="hand2")
        buttontrainbrowse.place(x=500,y=125,width=60,height=30)
        buttontrainbrowse["command"] = self.browse_fileradio

        #train buttons
        ft = tkFont.Font(family='Times',size=9)
        self.buttontrain=tk.Button(root,font = ft, text = "Load", cursor="hand2")
        self.buttontrain.place(x=510,y=175,width=50,height=30)
        self.buttontrain["command"] = self.load

        self.buttonsave=tk.Button(root,font = ft, text = "Save Model", state=DISABLED)
        self.buttonsave.place(x=415,y=175,width=85,height=30)
        self.buttonsave["command"] = self.savemodel

        self.buttontel=tk.Button(root,font = ft, text = "Model Analysis", state=DISABLED)
        self.buttontel.place(x=295,y=175,width=110,height=30)
        self.buttontel["command"] = self.analysis

        labelmoddis = tk.Label(root,font = tkFont.Font(family='Times',size=10), text="Model :", justify = LEFT)
        labelmoddis.place(x=30, y=180)
        self.labelmod = tk.Label(root,font = tkFont.Font(family='Times',size=10), text='Nil', fg='black', justify = LEFT)
        self.labelmod.place(x=90, y=180)

        #Get data Path
        ft = tkFont.Font(family='Times',size=10)
        self.label3=tk.Label(root, font = ft,justify = LEFT,text = "Scan Data : ")
        self.label3.place(x=30,y=240)

        self.entryscan=tk.Entry(root, font = ft, justify = LEFT, textvariable = self.scanloc)
        self.entryscan.place(x=150,y=240,width=350,height=30)

        self.labelinvalidscan = tk.Label(root,font = tkFont.Font(family='Times',size=9), justify = LEFT)
        self.labelinvalidscan.place(x=150, y=270)

        buttonscanbrowse=tk.Button(root, font = ft, text = "Browse", cursor="hand2")
        buttonscanbrowse.place(x=500,y=240,width=60,height=30)
        buttonscanbrowse["command"] = self.browse_file
        
        labelpreddis = tk.Label(root,font = tkFont.Font(family='Times',size=10), text="Prediction :", justify = LEFT)
        labelpreddis.place(x=30, y=300)
        self.labelpred = tk.Label(root,font = tkFont.Font(family='Times',size=18), justify = LEFT)
        self.labelpred.place(x=130, y=290)

        buttonpredict=tk.Button(root,font = ft, text = "Predict")
        buttonpredict.place(x=400,y=290,width=90,height=30)
        buttonpredict["command"] = self.predict

    def train(self):
        run=False
        try:
            path = str(self.trainloc.get())
            dat = pd.read_csv(path)
            y = dat.loc[:, 'target']
            X = dat.iloc[:, dat.columns!='target']
            X = preprocess(X)
            x_train, x_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model=train(x_train, y_train, x_test, self.y_test)
            self.y_pred=self.model.predict(x_test)
            self.labelmod.config(text=str(self.model), fg='green')
            self.buttonsave.config(state=NORMAL, cursor='hand2')
            self.buttontel.config(state=NORMAL, cursor='hand2')
            self.labelinvalidtrain.config(text="")
            messagebox.showinfo("Successful", "Model Trained Successfully")
            run=True
        except FileNotFoundError:
            self.labelinvalidtrain.config(text="*invalid path", fg='red')
        except KeyError:
            messagebox.showerror("Failed", "Training Model Failed\nPlease check dataset format. \nThe model takes 30 feature columns and 1 target column")
        except Exception as e:
            if str(e)=='target':
                error="Training Model Failed\nError : No Target found. Data not Labelled"
            else:
                error="Training Model Failed\nError : " + str(e)
            messagebox.showerror("Failed", error)
        if not run:
            self.labelmod.config(text="Error", fg='red')
            self.model=None
    
    def load(self):
        run=False
        try:
            path = str(self.trainloc.get())
            with open(path, 'rb') as file:
                self.model=joblib.load(file)
                self.labelmod.config(text=str(self.model), fg='green')
            self.labelinvalidtrain.config(text="")
            run=True
        except FileNotFoundError:
            self.labelinvalidtrain.config(text="*invalid path", fg='red')
        except (EOFError, KeyError):
            messagebox.showerror("Failed", "Loading Model Failed!\nFile Corrupt or Wrong format")
        except Exception as e:
            error="Loading Model Failed\nError : " + str(e)
            messagebox.showerror("Failed", error)
        if not run:
            self.labelmod.config(text="Error", fg='red')
            self.model=None

    def predict(self):
        try:
            if not self.model:
                messagebox.showerror("Failed", "Model Not Found!\nPlease Load a Model File or Use the Train Model Function")
            else:
                path=str(self.scanloc.get())
                dat = pd.read_csv(path).iloc[:, :30]
                out=self.model.predict(dat)
                if out[0]==0:
                    self.labelpred.config(text="BENIGN", fg='green')
                else:
                    self.labelpred.config(text="MALIGNANT", fg='red')
            self.labelinvalidscan.config(text="")
        except FileNotFoundError:
            self.labelinvalidscan.config(text="*invalid path", fg='red')
        except ValueError:
            messagebox.showerror("Failed", "Invalid Format. The model takes 30 feature columns as input")
        except Exception as e:
            error="Operation Failed\nError : " + str(e)
            messagebox.showerror("Failed", error)

    def trainvisible(self, enable):
        if enable:
            self.buttontrain.config(text="Train", command=self.train)
        else:
            self.buttontrain.config(text="Load", command=self.load)
            self.buttonsave.config(state=DISABLED, cursor='arrow')
            self.buttontel.config(state=DISABLED, cursor='arrow')

    def savemodel(self):
        file_path = filedialog.asksaveasfilename(defaultextension='.joblib',filetypes=[('JOBLIB models', '*.joblib'),('All Files', '*.*')],title='Save File')
        if file_path:
            try:
                with open(file_path, 'wb') as file:
                    joblib.dump(self.model, file)
                    messagebox.showinfo("Successful", "Model Saved Successfully!")
            except:messagebox.showerror("Failed", "Saving Model Failed")

    def analysis(self):
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        fscore = f1_score(self.y_test, self.y_pred)
        confmatrix = confusion_matrix(self.y_test, self.y_pred)
        meansquareerror = mean_squared_error(self.y_test, self.y_pred)
        fpr, tpr, threshold = roc_curve(self.y_test, self.y_pred)
        aucscore=auc(fpr, tpr)
        info=f"Model Scores :\n\nAccuracy = {accuracy}\nPrecision = {precision}\nRecall = {recall}\nF-Score = {fscore}\nMeanSquareError = {meansquareerror}\nAUC Score = {aucscore}"

        window=Tk()
        window.title("Analysis")
        window.configure(bg ="white")
        width=500
        height=900
        screenwidth = window.winfo_screenwidth()
        screenheight = window.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        window.geometry(alignstr)
        window.resizable(width=False, height=False)
        
        figure = plt.Figure(figsize=(6,5), dpi=100)
        axes = figure.add_subplot(111)
        chart_type = FigureCanvasTkAgg(figure, window)
        chart_type.get_tk_widget().pack()
        toolbar = NavigationToolbar2Tk(chart_type,window)
        toolbar.update()
        caxes = axes.matshow(confmatrix, interpolation ='nearest', cmap='plasma')
        figure.colorbar(caxes)
        axes.set_title('Confusion Matrix')
        Label(window,font = tkFont.Font(family='Times',size=10), bg='white', text=info, justify = LEFT).place(x=30, y=600)
        window.mainloop()

    def browse_fileradio(self):
        if self.mod.get()==0:
            file_path = filedialog.askopenfilename(filetypes=[("JOBLIB Files", "*.joblib")])
        else:
            file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        self.trainloc.set(file_path) 

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        self.scanloc.set(file_path) 


def main():
    root = tk.Tk()
    gui = MainWindow(root)  
    root.mainloop()

if __name__ == '__main__':
    main()