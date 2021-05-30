import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import streamlit.components.v1 as components
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import sweetviz as sv
import codecs
import streamlit as st
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn import model_selection


def st_display_sweetviz(report_html,width=1000,height=500):
    report_file = codecs.open(report_html,'r')
    page = report_file.read()
    components.html(page,width=width,height=height,scrolling=True)
def main():
    """ML Dataset Explorer"""
    st.title("ML Dataset with Streamlit")
    st.subheader("Simple Data Science Explorer with Streamlit")
    html_temp = """
            <div style="background-color:tomato;padding:0px;border-radius:0px">
            <h1 style="color:white;text-align:center;">Abijith EDA & Machine leaninng application</h1>
            <h2 style="color:white;text-align:center;">Click the side button to view the Automated EDA</h1>
            <h3 style="color:white;text-align:center;">https://share.streamlit.io/abijith4/data-visualization-app/app.py</h1>
            </div>
            """

    menu=["Menu","Sweetviz"]
    choice=st.sidebar.selectbox("Menu",menu)
    if choice=="Sweetviz":
        st.subheader("Automated EDA with Sweetviz")
        data_filee = st.file_uploader("Upload CSV", type=['csv'])
        if data_filee is not None:
            df1 = pd.read_csv(data_filee)
            st.dataframe(df1.head())
            if st.button("Generate Sweetviz Report"):\
            report=sv.analyze(df1)
            report.show_html()
            st_display_sweetviz("SWEETVIZ_REPORT.html")




    components.html(html_temp)
    data_file = st.file_uploader("Upload CSV", type=['csv'])
    df = pd.read_csv(data_file)

    if st.checkbox("show dataset"):
        number=st.number_input("Number of Rows to view")
        st.dataframe(df.head())

    if  st.button("Column Names"):
        st.write(df.columns)

    if st.checkbox("Shape of Dataset"):
        data_dim=st.radio("Show Dimension By",("Rows","Columns"))
        if data_dim=='Rows':
            st.text("Number of Rows")
            st.write(df.shape[0])
        elif data_dim=='Columns':
            st.text("Number of Columns")
            st.write(df.shape[1])
        else:
            st.write(df.shape)

#columns
    if st.checkbox("Select columns to show"):
        all_columns=df.columns.tolist()
        select_columns=st.multiselect("Select",all_columns)
        new_df=df[select_columns]
        st.dataframe(new_df)

    if st.button("Value Counts"):
        st.text("Value Count By Target/Class")
        st.write(df.iloc[:,-1].value_counts())

    if st.button("Data Types"):
        st.write(df.dtypes)

    if st.button("Summary"):
        st.write(df.describe())

    if st.button("Missing values"):
        st.write(df.isnull())

    st.subheader("Data visualization")
    #core
    #seaborn
#corre
    if st.checkbox("Correlation plot [Seaborn]"):
        st.write(sns.heatmap(df.corr(),annot=True))
        st.pyplot()

    #countplot
    if st.checkbox("plot of value count"):
        st.text("Value Counts By Target")
        all_columns_names = df.columns.tolist()
        primary_col=st.selectbox("Primary column to GroupBy",all_columns_names)
        selected_column_names=st.multiselect("Select columns",all_columns_names)
        if st.button("plot"):
            st.text("Generating plot")
            if selected_column_names:
                vc_plot=df.groupby(primary_col)[selected_column_names].count()
            else:
                vc_plot = df.iloc[:,-1].value_counts()
            st.write(vc_plot.plot(kind="bar"))
            st.pyplot()


    #piecha
    if st.checkbox("pie plot"):
        all_columns_names = df.columns.tolist()
        st.info("Please Choose Target Column")
        int_column =  st.selectbox('Select Int Columns For Pie Plot',all_columns_names)
        if st.button("Generate Pie Plot"):
            cust_values = df[int_column].value_counts()
            st.write(cust_values.plot.pie(autopct="%1.1f%%"))
            st.success("Generating A Customizable Pie Plot")
            st.write(df.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
            st.pyplot()



    st.subheader("Customizable Plots")
    all_columns_names = df.columns.tolist()
    type_of_plot = st.selectbox("Select the Type of Plot",["area","bar","line","hist","box","kde"])
    selected_column_names = st.multiselect('Select Columns To Plot',all_columns_names)
    cust_target = df.iloc[:,-1].name

#*********************************************
    plot_fig_height = st.number_input("Choose Fig Size For Height",10,50)
    plot_fig_width = st.number_input("Choose Fig Size For Width",10,50)


#*************************************************

    if st.button("Generate Plot"):
        st.success("Generating A Customizable Plot of: {} for :: {}".format(type_of_plot,selected_column_names))
        if type_of_plot == 'area':
            cust_data = df[selected_column_names]
            st.area_chart(cust_data)
        elif type_of_plot == 'bar':
            cust_data = df[selected_column_names]
            st.bar_chart(cust_data)
        elif type_of_plot == 'line':
            cust_data = df[selected_column_names]
            st.line_chart(cust_data)
        elif type_of_plot == 'hist':
            custom_plot = df[selected_column_names].plot(kind=type_of_plot,bins=2)
            st.write(custom_plot)
            st.pyplot()
        elif type_of_plot == 'box':
            custom_plot = df[selected_column_names].plot(kind=type_of_plot)
            st.write(custom_plot)
            st.pyplot()
        elif type_of_plot == 'kde':
            custom_plot = df[selected_column_names].plot(kind=type_of_plot)
            st.write(custom_plot)
            st.pyplot()
        else:
            cust_plot = df[selected_column_names].plot(kind=type_of_plot)
            st.write(cust_plot)
            st.pyplot()


    st.subheader("Our Features and Target")
    if st.checkbox("Show Features"):
        all_features = df.iloc[:,0:-1]
        st.text('Features Names:: {}'.format(all_features.columns[0:-1]))
        st.dataframe(all_features.head())

    if st.checkbox("Show Target"):
        all_target = df.iloc[:,-1]
        st.text('Target/Class Name:: {}'.format(all_target.name))
        st.dataframe(all_target.head(10))



        x=df.iloc[:,0:-1]
        y=df.iloc[:,-1]


        st.markdown('**1.2. Dataset dimension**')
        st.write('x')
        st.info(x.shape)
        st.write('y')
        st.info(y.shape)

        st.markdown('**1.3. Variable details**:')
        st.write('x variable (first 20 are shown)')
        st.info(list(x.columns[:20]))
        st.write('y variable')
        st.info(y.name)

        with st.sidebar.header('2. Set Parameters'):
            split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
            seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)

        def filedownload(df, filename):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
            href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
            return href

        def imagedownload(plt, filename):
            s = io.BytesIO()
            plt.savefig(s, format='pdf', bbox_inches='tight')
            plt.close()
            b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
            href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
            return href



        x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = split_size,random_state = seed_number)
        reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
        models_train,predictions_train = reg.fit(x_train, x_train, y_train, y_train)
        models_test,predictions_test = reg.fit(x_train, x_test, y_train, y_test)

        st.subheader('2. Table of Model Performance')
        st.write(predictions_train)
        st.markdown(filedownload(predictions_train,'training.csv'), unsafe_allow_html=True)



        st.write('Test set')
        st.write(predictions_test)
        st.markdown(filedownload(predictions_test,'test.csv'), unsafe_allow_html=True)

        alg = ['Decision Tree', 'Support Vector Machine']
        classifier = st.selectbox('Which algorithm?', alg)
        if classifier=='Decision Tree':
            dtc = DecisionTreeClassifier()
            dtc.fit(x_train, y_train)
            acc = dtc.score(x_test, y_test)
            st.write('Accuracy: ', acc)
            pred_dtc = dtc.predict(x_test)
            cm_dtc=confusion_matrix(y_test,pred_dtc)
            st.write('Confusion matrix: ', cm_dtc)
        elif classifier == 'Support Vector Machine':
            svm=SVC()
            svm.fit(x_train, y_train)
            acc = svm.score(x_test, y_test)
            st.write('Accuracy: ', acc)
            pred_svm = svm.predict(x_test)
            cm=confusion_matrix(y_test,pred_svm)
            st.write('Confusion matrix: ', cm)
if __name__=='__main__':
    main()
