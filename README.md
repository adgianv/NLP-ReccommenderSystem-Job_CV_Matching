# Resume Matching Tool

This repository contains a comprehensive approach to matching resumes with job descriptions using natural language processing (NLP) techniques. The goal of this project is to automate the process of suggesting suitable job titles based on resume content, improving upon traditional keyword-matching methods commonly found in job search platforms.

## Introduction

Job recommender systems are critical tools for job seekers, yet existing methods often rely on simplistic keyword matching. This project focuses on advancing the state-of-the-art in job matching using NLP and machine learning techniques. The primary objective is to develop a model that can analyze a resume and recommend the most suitable job title based on its content.

## Approach

The project encompasses several key components:

1. **Literature Review**: Investigates existing methodologies in resume parsing and job matching, highlighting the use of NLP tools like cosine similarity, TF-IDF, and advanced embedding techniques.
  
2. **Data Collection and Preprocessing**:
   - **Resumes**: Retrieved from Kaggle and categorized into technical job titles.
   - **Job Descriptions**: Scraped from Indeed.com for 12 specified job titles.
   
3. **Exploratory Data Analysis**: Examines the distribution and common terms within the resume and job description datasets.

4. **Matching Methodology**:
   - **Data Preprocessing**: Includes text normalization, skill extraction using spaCy's Named Entity Recognition (NER), and TF-IDF vectorization.
   - **Models Tested**:
     - Cosine Similarity (Mode and Average)
     - Logistic Regression
     - Random Forest

5. **Evaluation**: Measures model performance using accuracy, recall, precision, and F1-score metrics.

6. **Interactive Tool Development**:
   - Implements a tool for BSE students to upload resumes and receive suggested job titles, along with identified missing skills.

## Results and Conclusion

The project demonstrates that leveraging advanced machine learning models like logistic regression and random forest significantly improves resume matching accuracy over traditional cosine similarity methods. The findings underscore the effectiveness of NLP techniques in automating the job matching process.

## Repository Structure

- `data/`: Contains datasets used for training and evaluation.
- `notebooks/`: Jupyter notebooks detailing data preprocessing, model training, and evaluation.
- `src/`: Source code for the interactive resume matching tool.
- `README.md`: Overview of the project, methodology, and key findings.
- `LICENSE`: License information for the repository.

## Getting Started

To explore the project:
1. Clone the repository.
2. Review the Jupyter notebooks in the `notebooks/` directory for detailed steps.
3. Run the interactive tool in the `src/` directory to match resumes with job titles.

For further details and usage instructions, refer to the complete report and project documentation.

---

This repository serves as a comprehensive exploration of NLP techniques for resume matching, offering insights into effective methodologies for job recommendation systems. The interactive tool developed as part of this project provides practical utility for BSE students seeking tailored job suggestions based on their resumes.
