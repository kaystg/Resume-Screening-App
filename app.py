import streamlit as st
import re
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
#loading models
clf=pickle.load(open('clf.pkl','rb'))
tfidf=pickle.load(open('tfidf (1).pkl','rb'))
#web app
#import re #regular expression used for pattern matching and text manipulation
def cleanresume(txt):
  cleantxt=re.sub('http\S+\S',' ',txt)#the space is the middle represents the replacement of the ommited text
  cleantxt=re.sub('RT|CC',' ',cleantxt)
  cleantxt=re.sub('@\S+',' ',cleantxt)
  cleantxt=re.sub('#\S+',' ',cleantxt)
  #now clearing the stopwords
  cleantxt=re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),' ',cleantxt)
  cleantxt=re.sub(r'[^\x00-\x7f]',' ',cleantxt)
  cleantxt=re.sub('\s+',' ',cleantxt)
  return cleantxt
def main():
    st.title("Resume Screening App")
    upload_file=st.file_uploader("Upload Resume",type=['txt','pdf'])
    if upload_file is not None:
        try:
            resume_bytes=upload_file.read()#it reades in bytes
            resume_text=resume_bytes.decode('utf-8')#decode the bytes to texts
        except UnicodeDecodeError:
            #If UTF-8 decoding fils,try decoding with 'Latin-1'
            resume_text=resume_bytes.decode('latin-1')
        cleaned_resume=cleanresume(resume_text)
        cleaned_resume=tfidf.transform([cleaned_resume])
        prediction_id=clf.predict(cleaned_resume)[0]
        st.write(prediction_id)
        #Map Category ID to categoru names
        category_mapping={
            15:"Java Developer",
            23:"Testing",
            8:"DevOps Engineer",
            20:"Python Developer",
            24:"Web Designer",
            12:"HR",
            13:"Hadoop",
            3:"Blockchain",
            10:"ETL Developer",
            18:"Operations Manager",
            6:"Data Science",
            22:"Sales",
            16:"Mechanical Engineer",
            1:"Arts",
            7:"Database",
            11:"Electrical Engineering",
            14:"Health and fitness",
            19:"PMO",
            4:"Business Analyst",
            9:"DotNet Developer",
            2:"Automation Testing",
            17:"Network Security Engineer",
            21:"SAP Developer",
            5:"Civil Engineer",
            0:"Advpcate",
        }
        category_name=category_mapping.get(prediction_id,"Unknown")
        st.write("Predicted Category:",category_name )
#python main
if __name__=="__main__":
    main()
