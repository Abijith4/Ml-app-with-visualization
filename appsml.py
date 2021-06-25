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
    Visualization =["Correlation plot","Pair plot","Histogram","Box Plot","pie plot","Mean","Median","Mode"]



    choice=st.sidebar.selectbox("Visualization & other function",Visualization)
    if choice=="Correlation plot [Seaborn]":
        st.write(sns.heatmap(df.corr(),annot=True))
        st.pyplot()

    if choice=="Pair plot":
        hue = st.selectbox("Please specify a hue",df.columns)
        fig = plt.figure()
        fig=sns.pairplot(df,hue=hue)
        #st.plotly_chart(fig)
        st.pyplot(fig)

    if choice =="Histogram":
        hue = st.selectbox("Please specify a hue",df.columns)
        a = st.selectbox("Please Specify a name attribute, the name will be used to label the data axis.",df.columns)
        fig = plt.figure()
        fig = sns.FacetGrid(df, hue=hue) \
            .map(sns.distplot, a)\
            .add_legend();
        st.pyplot(fig)


    if choice =="Box Plot":
        fig,ax=plt.subplots()
        ax=df.boxplot()

        st.pyplot(fig)
        #optionm=["Mean","Median","Mode","K NN Imputer"]


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

    if choice =="Mean":
        clean_df=(df.fillna(df.mean()))
        clean_df.fillna(clean_df.select_dtypes(include='object').mode().iloc[0], inplace=True)
        st.dataframe(clean_df)
        st.write(clean_df.dtypes)
        st.info("The Percenatge of Value Missing in Given Data is : {:.2f}%".format(((df.isna().sum().sum())/(df.count().sum())*100)))
        st.info("Data to be treated using MEAN : {}".format(list(dict(df.mean()).keys())))
        st.info('Shape of dataframe (Rows, Columns):{} '.format(df.shape))
        st.write('Data description : ',df.describe())
        st.info("Only Numerical missing values will be treated using MEAN ")
        st.info("categorical missing values will be treated using MODE ")
        st.write("\nEmpty rows  after imputing the data: \n", clean_df.isnull().sum())
        st.line_chart(clean_df)

    if choice =="Median":
        clean_df=(df.fillna(df.median()))
        clean_df.fillna(clean_df.select_dtypes(include='object').mode().iloc[0], inplace=True)
        st.dataframe(clean_df)
        st.write(df.dtypes)
        st.info("The Percenatge of Value Missing in Given Data is : {:.2f}%".format(((df.isna().sum().sum())/(df.count().sum())*100)))
        st.info("Data to be treated using MEDIAN : {}".format(list(dict(df.median()).keys())))
        st.info('Shape of dataframe (Rows, Columns):{} '.format(df.shape))
        st.write('Data description : ',df.describe())
        st.info("Only Numerical missing values will be treated using Median ")
        st.info("categorical missing values will be treated using MODE ")
        st.write("\nEmpty rows  after imputing the data: \n", clean_df.isnull().sum())
        st.line_chart(clean_df)


    if choice =="Mode":
        cat_col=list(df.select_dtypes(include ='object').columns)
        st.info("The Percentage of Value Missing in Given Data is : {:.3f}%".format((df[cat_col].isna().sum().sum())/(df.count().sum())*100))
        st.info("\nThe Percenatge of Value Missing in Given Data is :\n{}".format((df[cat_col].isnull().sum()*100)/df.shape[0]))
        clean_df=(df.fillna(df.select_dtypes(include ='object').mode().iloc[0]))
        st.dataframe(clean_df)
        st.info("\nData to be treated using MODE : {}".format(cat_col))
        st.write('Shape of dataframe (Rows, Columns): ',df.shape)
        st.write('Data description :\n',df.describe(include ='object'))
        st.info("Only categorical missing values will be treated using MODE ")
        st.write("\nEmpty rows  after imputing the data: \n", clean_df.isnull().sum())
        st.info("You can head to Mean or Median to treat the Numerical Missing Value")
        st.line_chart(clean_df)





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
