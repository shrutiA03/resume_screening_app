# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 21:08:53 2025

@author: SHRUTI-NIDHI
"""

import re
import streamlit as st
import spacy
from spacy.matcher import Matcher
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize spaCy and sentence transformer
nlp = spacy.load('en_core_web_sm')
embedder = SentenceTransformer('all-MiniLM-L6-v2')
stop_words = set(stopwords.words('english'))

def extract_text_from_pdf(pdf_file):
    """Extract text from an uploaded PDF file."""
    try:
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.read())
        text = extract_text("temp.pdf")
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def preprocess_text(text):
    """Tokenization, stopword removal, and lemmatization."""
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words and token.isalnum()]
    doc = nlp(" ".join(tokens))
    lemmatized = [token.lemma_ for token in doc]
    return lemmatized

def extract_contact_number(text):
    """Extract phone number using regex."""
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    return match.group() if match else None

def extract_email(text):
    """Extract email using regex."""
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    return match.group() if match else None

def extract_name(text):
    """Extract name using NER and POS tagging."""
    doc = nlp(text[:1000])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    matcher = Matcher(nlp.vocab)
    patterns = [
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}],
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}]
    ]
    for pattern in patterns:
        matcher.add('NAME', [pattern])
    matches = matcher(doc)
    for _, start, end in matches:
        return doc[start:end].text
    return None

def extract_skills(text, skills_list):
    """Extract skills using regex and semantic matching."""
    preprocessed_text = preprocess_text(text)
    found_skills = []
    for skill in skills_list:
        if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE):
            found_skills.append(skill)
        else:
            skill_embedding = embedder.encode(skill.lower())
            text_embedding = embedder.encode(" ".join(preprocessed_text))
            similarity = util.cos_sim(skill_embedding, text_embedding).item()
            if similarity > 0.7:
                found_skills.append(skill)
    return list(set(found_skills)) if found_skills else None

def extract_education(text):
    """Extract education using regex and NER."""
    pattern = r"(?i)(B\s?Tech|B\.Tech|BTech|Bsc|M\s?Tech|M\.Tech|MTech|\bB\.\w+|\bM\.\w+|\bPh\.D\.|\bBachelor(?:'s)?|\bMaster(?:'s)?|\bPh\.D)\s*([^\n]*)"
    matches = re.findall(pattern, text)
    education = [f"{degree} {details}".strip() for degree, details in matches if degree]
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "ORG" and any(keyword in ent.text.lower() for keyword in ["university", "college", "institute"]):
            education.append(ent.text)
    return list(set(education)) if education else None

def extract_job_roles(text):
    """Extract job roles using POS tagging and NER."""
    doc = nlp(text)
    job_roles = []
    for sent in doc.sents:
        for token in sent:
            if token.pos_ == "NOUN" and token.dep_ in ["nsubj", "dobj"]:
                if any(keyword in token.text.lower() for keyword in ["engineer", "manager", "analyst", "developer"]):
                    job_roles.append(token.text)
    return list(set(job_roles)) if job_roles else None

def analyze_job_description(jd_text, skills_list):
    """Analyze job description for required skills and qualifications."""
    preprocessed_jd = preprocess_text(jd_text)
    required_skills = extract_skills(jd_text, skills_list)
    qualifications = extract_education(jd_text)
    return {
        "skills": required_skills or [],
        "qualifications": qualifications or [],
        "preprocessed": preprocessed_jd
    }

def skills_gap_analysis(resume_skills, jd_skills):
    """Perform skills gap analysis."""
    resume_skills = set(resume_skills or [])
    jd_skills = set(jd_skills or [])
    matching_skills = resume_skills.intersection(jd_skills)
    missing_skills = jd_skills - resume_skills
    return {
        "matching_skills": list(matching_skills),
        "missing_skills": list(missing_skills)
    }

def calculate_ats_score(resume_text, jd_text, resume_skills, jd_skills):
    """Calculate ATS score based on keyword matching and semantic similarity."""
    resume_keywords = set(preprocess_text(resume_text))
    jd_keywords = set(preprocess_text(jd_text))
    common_keywords = resume_keywords.intersection(jd_keywords)
    keyword_score = (len(common_keywords) / len(jd_keywords)) * 50 if jd_keywords else 0

    resume_skill_set = set(resume_skills or [])
    jd_skill_set = set(jd_skills or [])
    skill_match_score = (len(resume_skill_set.intersection(jd_skill_set)) / len(jd_skill_set)) * 30 if jd_skill_set else 0

    resume_embedding = embedder.encode(resume_text[:1000])
    jd_embedding = embedder.encode(jd_text[:1000])
    semantic_score = util.cos_sim(resume_embedding, jd_embedding).item() * 20

    total_score = keyword_score + skill_match_score + semantic_score
    return round(total_score, 2)

def generate_feedback(gap_analysis, ats_score, resume_education, jd_qualifications):
    """Generate feedback based on analysis."""
    feedback = []
    if gap_analysis["missing_skills"]:
        feedback.append(f"Consider acquiring the following skills: {', '.join(gap_analysis['missing_skills'])}.")
    if not resume_education:
        feedback.append("Add education details to your resume.")
    elif jd_qualifications and not any(qual in resume_education for qual in jd_qualifications):
        feedback.append("Your education may not fully match the job's required qualifications.")
    if ats_score < 60:
        feedback.append("Improve keyword usage to align with the job description for better ATS compatibility.")
    elif ats_score < 80:
        feedback.append("Your resume is a good match but could benefit from minor tweaks to keywords and skills.")
    else:
        feedback.append("Your resume aligns well with the job description!")
    return feedback

def create_dashboard(resume_skills, jd_skills, ats_score, gap_analysis):
    """Create a dashboard with analytics."""
    st.subheader("Resume Analytics Dashboard")
    
    # ATS Score
    st.metric("ATS Score", f"{ats_score}%")
    
    # Skills Comparison
    st.write("Skills Comparison")
    skills_data = {
        "Skill": resume_skills or [],
        "Source": ["Resume"] * len(resume_skills or [])
    }
    skills_data["Skill"].extend(jd_skills or [])
    skills_data["Source"].extend(["Job Description"] * len(jd_skills or []))
    skills_df = pd.DataFrame(skills_data)
    fig, ax = plt.subplots()
    sns.countplot(data=skills_df, x="Skill", hue="Source", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Gap Analysis
    st.write("Skills Gap Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Matching Skills", gap_analysis["matching_skills"])
    with col2:
        st.write("Missing Skills", gap_analysis["missing_skills"])

# Streamlit App
def main():
    st.title("Resume Parser and Job Match Analyzer")
    
    # Skills List
    skills_list = [
        'Python', 'Data Analysis', 'Machine Learning', 'Communication',
        'Project Management', 'Deep Learning', 'SQL', 'Tableau',
        'Java', 'C++', 'JavaScript', 'AWS', 'Docker', 'Kubernetes'
    ]
    
    # Resume Upload
    st.header("Upload Resume")
    resume_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
    resume_text = ""
    if resume_file:
        resume_text = extract_text_from_pdf(resume_file)
        st.success("Resume uploaded successfully!")
    
    # Job Description Input
    st.header("Enter Job Description")
    jd_text = st.text_area("Paste the job description here")
    
    if resume_text and jd_text and st.button("Analyze"):
        # Resume Parsing
        name = extract_name(resume_text)
        contact = extract_contact_number(resume_text)
        email = extract_email(resume_text)
        resume_skills = extract_skills(resume_text, skills_list)
        education = extract_education(resume_text)
        job_roles = extract_job_roles(resume_text)
        
        st.header("Resume Details")
        st.write(f"*Name*: {name or 'Not found'}")
        st.write(f"*Contact*: {contact or 'Not found'}")
        st.write(f"*Email*: {email or 'Not found'}")
        st.write(f"*Skills*: {', '.join(resume_skills) if resume_skills else 'Not found'}")
        st.write(f"*Education*: {', '.join(education) if education else 'Not found'}")
        st.write(f"*Job Roles*: {', '.join(job_roles) if job_roles else 'Not found'}")
        
        # Job Description Analysis
        jd_analysis = analyze_job_description(jd_text, skills_list)
        st.header("Job Description Analysis")
        st.write(f"*Required Skills*: {', '.join(jd_analysis['skills']) if jd_analysis['skills'] else 'Not found'}")
        st.write(f"*Qualifications*: {', '.join(jd_analysis['qualifications']) if jd_analysis['qualifications'] else 'Not found'}")
        
        # Skills Gap Analysis
        gap_analysis = skills_gap_analysis(resume_skills, jd_analysis["skills"])
        
        # ATS Score
        ats_score = calculate_ats_score(resume_text, jd_text, resume_skills, jd_analysis["skills"])
        
        # Feedback
        feedback = generate_feedback(gap_analysis, ats_score, education, jd_analysis["qualifications"])
        st.header("Feedback")
        for item in feedback:
            st.write(f"- {item}")
        
        # Dashboard
        create_dashboard(resume_skills, jd_analysis["skills"], ats_score, gap_analysis)
if __name__ == "__main__":

    main()
