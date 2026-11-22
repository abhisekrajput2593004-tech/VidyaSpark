import os
import time
import urllib.parse
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="VidyaSpark | India's Smartest Learning Hub",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# COLORFUL PRO CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .hero-banner {
        background: linear-gradient(135deg, #2563EB 0%, #7C3AED 50%, #EC4899 100%);
        padding: 1.6rem 2rem; border-radius: 16px; color: white; margin-bottom: 1.2rem;
        box-shadow: 0 10px 25px rgba(124, 58, 237, 0.35);
    }
    .hero-banner h1 { margin: 0; font-size: 2.2rem; font-weight: 800; color: #FFFFFF; }
    .hero-banner p { margin: 0.35rem 0 0 0; font-size: 1rem; color: #F3E8FF; }
    .hero-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #A855F7, #F43F5E);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.15rem;
    }
    .hero-subtitle { font-size: 0.98rem; color: #CBD5E1; margin-bottom: 1.1rem; }
    .sidebar-brand {
        background: linear-gradient(135deg, #3B82F6 0%, #9333EA 100%);
        padding: 0.9rem; border-radius: 12px; text-align: center; color: white; margin-bottom: 0.8rem;
    }
    .sidebar-brand h2 { margin: 0; font-size: 1.45rem; font-weight: 800; color: #FFFFFF; }
    .sidebar-brand p { margin: 0.2rem 0 0 0; font-size: 0.76rem; color: #E0E7FF; }
    .kpi-card {
        border-radius: 14px; padding: 1.1rem; text-align: center; color: #FFFFFF;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.28); transition: transform 0.2s ease;
    }
    .kpi-card:hover { transform: translateY(-4px); }
    .kpi-blue { background: linear-gradient(135deg, #2563EB 0%, #06B6D4 100%); }
    .kpi-orange { background: linear-gradient(135deg, #F97316 0%, #EF4444 100%); }
    .kpi-purple { background: linear-gradient(135deg, #7C3AED 0%, #EC4899 100%); }
    .kpi-green { background: linear-gradient(135deg, #10B981 0%, #059669 100%); }
    .kpi-value { font-size: 1.9rem; font-weight: 800; color: #FFFFFF; }
    .kpi-label { font-size: 0.85rem; font-weight: 600; color: #F8FAFC; margin-top: 0.2rem; }
    .imp-box {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%);
        border-left: 5px solid #F43F5E; border-radius: 10px; padding: 0.85rem 1rem; margin-bottom: 0.65rem;
    }
    .ai-explain-box {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 100%);
        border: 1px solid #7C3AED; border-left: 5px solid #38BDF8;
        border-radius: 12px; padding: 1.1rem 1.3rem; margin-bottom: 1rem;
    }
    .badge {
        display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600; margin-right: 0.3rem; margin-bottom: 0.3rem;
    }
    .badge-imp { background-color: #991B1B; color: #FEE2E2; border: 1px solid #F87171; }
    .badge-beginner { background-color: #065F46; color: #A7F3D0; }
    .badge-intermediate { background-color: #92400E; color: #FDE68A; }
    .badge-advanced { background-color: #9F1239; color: #FECDD3; }
    .badge-lang { background-color: #6B21A8; color: #F3E8FF; }
    .badge-meta { background-color: #334155; color: #F1F5F9; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 10-STREAM ALL-INDIA DATASET
# ---------------------------------------------------------
INITIAL_DATA = [
    # 1. B.Tech CSE
    ["B.Tech CSE", "Python Programming", "Beginner", "YouTube Video", "Hindi", "CodeWithHarry", "10 hrs", True,
     "Python Tutorial For Beginners in Hindi", "Complete Python course in Hindi with notes and projects.", "https://www.youtube.com/watch?v=UrsmFxEIp5k"],
    ["B.Tech CSE", "Python Programming", "Beginner", "YouTube Video", "English", "FreeCodeCamp", "4.5 hrs", True,
     "Learn Python - Full Course for Beginners", "Complete Python basics covering loops, functions and OOP.", "https://www.youtube.com/watch?v=rfscVS0vtbw"],
    ["B.Tech CSE", "DSA (Data Structures)", "Intermediate", "Course", "Hindi", "Striver / takeUforward", "25 hrs", True,
     "Striver A2Z DSA Course & Sheet", "Top DSA roadmap covering arrays, trees, graphs and DP.", "https://takeuforward.org/strivers-a2z-dsa-course/strivers-a2z-dsa-course-sheet-2/"],
    ["B.Tech CSE", "CS Core (OS/DBMS/CN)", "Intermediate", "YouTube Video", "Hindi", "Gate Smashers", "20 hrs", True,
     "OS, DBMS, Computer Networks (Gate Smashers)", "Best university exam & GATE revision playlists.", "https://www.youtube.com/results?search_query=Gate+Smashers+DBMS+OS+CN"],
    ["B.Tech CSE", "Machine Learning & AI", "Intermediate", "YouTube Video", "Hindi", "CampusX", "30 hrs", True,
     "100 Days of Machine Learning (CampusX)", "Hindi ML, NLP, and Data Science tutorials with projects.", "https://www.youtube.com/results?search_query=CampusX+100+Days+of+Machine+Learning"],

    # 2. Mechanical
    ["Mechanical", "Thermodynamics", "Beginner", "Course", "English", "NPTEL / SWAYAM", "30 hrs", True,
     "Engineering Thermodynamics - NPTEL IIT", "Free IIT professor lectures on laws of thermodynamics & entropy.", "https://nptel.ac.in/courses"],
    ["Mechanical", "Fluid Mechanics", "Intermediate", "YouTube Video", "Hindi", "Gate Wallah (PW)", "12 hrs", True,
     "Fluid Mechanics Complete Revision (Gate Wallah)", "Bernoulli equation, laminar & turbulent flow for B.Tech & GATE.", "https://www.youtube.com/results?search_query=Gate+Wallah+Fluid+Mechanics+Mechanical+Hindi"],
    ["Mechanical", "Strength of Materials", "Intermediate", "Documentation", "English", "Sanfoundry", "Self-paced", True,
     "Strength of Materials Concepts & MCQs", "Stress, strain, bending moment and shear force practice.", "https://www.sanfoundry.com/strength-materials-questions-answers/"],

    # 3. Electrical
    ["Electrical", "Circuit & Network Theory", "Beginner", "YouTube Video", "English", "Neso Academy", "15 hrs", True,
     "Network Theory & Basic Electrical Engineering", "Complete KVL, KCL, Thevenin & AC circuit analysis.", "https://www.youtube.com/results?search_query=Neso+Academy+Network+Theory+Electrical"],
    ["Electrical", "Electrical Machines", "Intermediate", "YouTube Video", "Hindi", "Kreatryx (Ankit Goyal)", "18 hrs", True,
     "Transformers & DC Machines in Hindi", "Detailed Hindi lectures on transformers and induction motors.", "https://www.youtube.com/results?search_query=Ankit+Goyal+Kreatryx+Electrical+Machines+Hindi"],
    ["Electrical", "Power Systems", "Advanced", "Course", "English", "NPTEL / SWAYAM", "35 hrs", True,
     "Power System Analysis - NPTEL IIT", "Transmission lines, load flow and fault analysis.", "https://nptel.ac.in/courses"],

    # 4. Class 10th & 12th
    ["Class 10th & 12th", "NCERT All Subjects", "Beginner", "Documentation", "English", "NCERT Official", "Self-paced", True,
     "Official NCERT Textbooks (Class 10 & 12)", "Free official NCERT PDFs for Science, Maths, SST.", "https://ncert.nic.in/textbook.php"],
    ["Class 10th & 12th", "Physics & Chemistry Boards", "Beginner", "App / Course", "Hindi", "Physics Wallah (PW)", "20 hrs", True,
     "PW Class 10th & 12th Board One-Shot", "Complete CBSE & State Board chapter-wise revision with PYQs.", "https://www.pw.live/"],
    ["Class 10th & 12th", "Board Strategy & Revision", "Beginner", "YouTube Video", "Hindi", "Next Toppers", "10 hrs", True,
     "Next Toppers 10th & 12th Marathon", "High-scoring board exam notes and rapid revision.", "https://www.youtube.com/results?search_query=Next+Toppers+Class+10+12+One+Shot"],

    # 5. JEE
    ["JEE", "Complete JEE Prep", "Intermediate", "App / Course", "Hindi", "Physics Wallah (PW JEE)", "Ongoing", True,
     "PW JEE Batches & YouTube One-Shots", "Structured JEE Main & Advanced batches and DPPs.", "https://www.pw.live/study/batches"],
    ["JEE", "Physics (JEE Main/Adv)", "Advanced", "YouTube Video", "English", "Physics Galaxy (Ashish Arora)", "25 hrs", True,
     "Physics Galaxy Revision Checklist", "Gold-standard JEE Physics concepts and illustrations.", "https://www.youtube.com/results?search_query=Physics+Galaxy+Revision+Checklist+JEE"],
    ["JEE", "Mathematics (JEE)", "Advanced", "YouTube Video", "Hindi", "Mohit Tyagi & MathonGo", "30 hrs", True,
     "JEE Maths Full Syllabus & PYQ Analysis", "Complete JEE Advanced Maths playlists + chapter weightage.", "https://www.youtube.com/results?search_query=MathonGo+Mohit+Tyagi+JEE+Maths"],

    # 6. NEET
    ["NEET", "Complete NEET Prep", "Beginner", "App / Course", "Hindi", "Physics Wallah (PW NEET)", "Ongoing", True,
     "PW NEET Batches (MR Sir, Pankaj Sir)", "NCERT-based NEET lectures and daily practice sheets.", "https://www.pw.live/"],
    ["NEET", "NCERT Biology", "Beginner", "YouTube Video", "Hindi", "Garima Goel / Seep Pahuja", "20 hrs", True,
     "NCERT Line-by-Line Biology for NEET", "Complete Botany & Zoology NCERT mastery with MCQs.", "https://www.youtube.com/results?search_query=Seep+Pahuja+Garima+Goel+NEET+Biology+NCERT"],
    ["NEET", "NEET Test Series", "Intermediate", "App / Course", "English", "Allen Digital & Unacademy", "Ongoing", True,
     "Allen Digital & Unacademy NEET Test Series", "All-India mock tests and NCERT assertion-reason practice.", "https://allen.in/"],

    # 7. UPSC
    ["UPSC", "GS & Hindi Medium Prep", "Beginner", "App / Course", "Hindi", "Drishti IAS", "Ongoing", True,
     "Drishti IAS Notes & Current Affairs", "Best UPSC editorial analysis, Mains answer writing and GS.", "https://www.drishtiias.com/"],
    ["UPSC", "Current Affairs & GS", "Intermediate", "YouTube Video", "Hindi", "StudyIQ IAS", "15 hrs", True,
     "StudyIQ IAS Daily Newspaper & Polity", "Daily newspaper breakdown and Prelims/Mains playlists.", "https://www.studyiq.com/"],
    ["UPSC", "Previous Papers & Official", "Advanced", "Practice", "English", "UPSC Official & PIB", "Ongoing", True,
     "UPSC Official Previous Year Papers", "Download official Prelims & Mains PYQs.", "https://upsc.gov.in/examinations/previous-question-papers"],

    # 8. SSC & Govt Exams
    ["SSC & Govt Exams", "General Studies (GS)", "Beginner", "App / Course", "Hindi", "Khan Global Studies (Khan Sir)", "Ongoing", True,
     "Khan Sir GS Special (History, Polity, Science)", "Most popular Hindi GS classes for SSC, Police & State Exams.", "https://khanglobalstudies.com/"],
    ["SSC & Govt Exams", "Mock Tests & PYQs", "Intermediate", "Practice", "English", "Testbook", "Ongoing", True,
     "Testbook Pass - All-India Mock Tests", "India's #1 mock test platform for SSC, Police & State Exams.", "https://testbook.com/"],
    ["SSC & Govt Exams", "Quantitative Aptitude (Maths)", "Intermediate", "YouTube Video", "Hindi", "Aditya Ranjan / Gagan Pratap", "20 hrs", True,
     "SSC Maths Complete Arithmetic & Advance", "Shortcut tricks and chapter-wise PYQ practice.", "https://www.youtube.com/results?search_query=Aditya+Ranjan+Gagan+Pratap+SSC+Maths+Playlist"],

    # 9. Banking & Railway
    ["Banking & Railway", "Banking Awareness & Reasoning", "Beginner", "App / Course", "Hindi", "Adda247", "Ongoing", True,
     "Adda247 IBPS PO/Clerk, SBI & RRB Live Classes", "Daily live classes, puzzles, seating arrangement & banking awareness.", "https://www.adda247.com/"],
    ["Banking & Railway", "Railway Maths (RRB NTPC/ALP)", "Intermediate", "YouTube Video", "Hindi", "Quick Tricks by Sahil Sir", "20 hrs", True,
     "Railway Maths Complete Playlist (Sahil Sir)", "Best Hindi shortcut tricks and PYQs for RRB NTPC, ALP & Group D.", "https://www.youtube.com/results?search_query=Sahil+Sir+Railway+Maths+Complete+Playlist"],
    ["Banking & Railway", "Banking & Railway Mock Tests", "Advanced", "Practice", "English", "Oliveboard & Testbook", "Ongoing", True,
     "Oliveboard & Testbook Banking/Railway Mocks", "Exam-level Prelims & Mains mock tests with All-India percentile.", "https://www.oliveboard.in/"],
    ["Banking & Railway", "Railway General Science", "Beginner", "YouTube Video", "Hindi", "Khan Sir / MD Classes", "15 hrs", True,
     "Railway General Science NCERT Marathon", "High-weightage Physics, Chem & Bio MCQs for RRB Technician/NTPC.", "https://www.youtube.com/results?search_query=Railway+General+Science+NCERT+Hindi+Marathon"],

    # 10. SSB & Defence
    ["SSB & Defence", "NDA / CDS & SSB Prep", "Beginner", "App / Course", "Hindi", "Defence Wallah (PW)", "Ongoing", True,
     "Defence Wallah NDA, CDS & SSB Complete Prep", "Affordable live batches and free YouTube marathons.", "https://www.youtube.com/results?search_query=Defence+Wallah+NDA+CDS+SSB+Interview"],
    ["SSB & Defence", "SSB 5-Day Process & OLQs", "Intermediate", "YouTube Video", "Hindi", "Learn With Sumit (LWS)", "10 hrs", True,
     "SSB Screening (OIR/PPDT), Psych & GTO Guide", "Complete SSB interview prep covering TAT, WAT, SRT & Lecturette.", "https://www.youtube.com/results?search_query=Learn+With+Sumit+SSB+Interview+Complete+Guide"],
    ["SSB & Defence", "Official Guidelines", "Intermediate", "Article", "English", "Join Indian Army", "Self-paced", True,
     "Official SSB Guidelines & 15 OLQs", "Authentic notification updates and 15 Officer Like Qualities.", "https://joinindianarmy.nic.in/"],
]

COLUMNS = ["Stream", "Topic", "Difficulty", "Resource_Type", "Language", "Platform", "Duration", "Important", "Title", "Description", "Link"]
ALL_LANGUAGES = ["All Languages", "Hindi", "English", "Hinglish", "Punjabi", "Tamil", "Telugu", "Bengali", "Marathi", "Gujarati", "Kannada", "Malayalam"]

# ---------------------------------------------------------
# TIMED SAMPLE PAPERS (ALL 10 STREAMS)
# ---------------------------------------------------------
SAMPLE_PAPERS = {
    "B.Tech CSE": [
        ("Which sorting algorithm has O(n log n) average-case time complexity?", ["Bubble Sort", "Merge Sort", "Insertion Sort", "Selection Sort"], "Merge Sort"),
        ("In DBMS, which normal form removes transitive dependency?", ["1NF", "2NF", "3NF", "BCNF"], "3NF"),
        ("Which OS scheduling algorithm can cause starvation?", ["Round Robin", "FCFS", "Shortest Job First (SJF)", "None"], "Shortest Job First (SJF)"),
        ("Output of `len({1, 2, 2, 3})` in Python is:", ["4", "3", "2", "Error"], "3"),
        ("Which protocol operates at the Transport Layer?", ["HTTP", "IP", "TCP", "ARP"], "TCP"),
    ],
    "Mechanical": [
        ("Carnot cycle efficiency depends only on:", ["Working fluid", "Source and Sink temperatures", "Pressure ratio", "Volume"], "Source and Sink temperatures"),
        ("Reynolds number is the ratio of:", ["Inertia to Viscous force", "Viscous to Gravity force", "Pressure to Inertia force", "None"], "Inertia to Viscous force"),
        ("Hooke's law is valid up to:", ["Yield point", "Ultimate stress", "Proportional limit", "Fracture point"], "Proportional limit"),
        ("A 4-stroke engine completes one power stroke in how many crank revolutions?", ["1", "2", "4", "0.5"], "2"),
        ("SI unit of dynamic viscosity is:", ["N·s/m² (Pa·s)", "m²/s", "N/m", "J/kg"], "N·s/m² (Pa·s)"),
    ],
    "Electrical": [
        ("Kirchhoff's Current Law (KCL) is based on conservation of:", ["Energy", "Charge", "Momentum", "Flux"], "Charge"),
        ("A transformer works on the principle of:", ["Self induction", "Mutual induction", "Electrostatic induction", "Lorentz force"], "Mutual induction"),
        ("Power factor of a purely resistive AC circuit is:", ["0", "0.5", "1 (Unity)", "Infinity"], "1 (Unity)"),
        ("In a 3-phase star connection, Line Voltage (VL) equals:", ["Vph", "√3 * Vph", "Vph / √3", "3 * Vph"], "√3 * Vph"),
        ("Which logic gate outputs 1 only when all inputs are 1?", ["OR", "NAND", "AND", "XOR"], "AND"),
    ],
    "Class 10th & 12th": [
        ("SI unit of Electric Power is:", ["Joule", "Watt", "Volt", "Ohm"], "Watt"),
        ("Which gas is released during photosynthesis?", ["CO2", "Nitrogen", "Oxygen", "Hydrogen"], "Oxygen"),
        ("If tan(θ) = 1, then acute angle θ is:", ["30°", "45°", "60°", "90°"], "45°"),
        ("Focal length of a plane mirror is:", ["Zero", "10 cm", "Infinity", "-1 cm"], "Infinity"),
        ("Chemical formula of Baking Soda is:", ["Na2CO3", "NaHCO3", "NaCl", "CaCO3"], "NaHCO3"),
    ],
    "JEE": [
        ("At 45° projection angle, horizontal range is:", ["Minimum", "Maximum", "Zero", "Half"], "Maximum"),
        ("The value of ∫ (1/x) dx is:", ["x²/2 + C", "ln|x| + C", "e^x + C", "-1/x² + C"], "ln|x| + C"),
        ("Hybridization of Carbon in CH4 is:", ["sp", "sp2", "sp3", "sp3d"], "sp3"),
        ("Dimensional formula of Force is:", ["[MLT^-1]", "[MLT^-2]", "[ML^2T^-2]", "[MT^-2]"], "[MLT^-2]"),
        ("For an orthogonal matrix A, A * A^T equals:", ["Zero matrix", "Identity matrix (I)", "2I", "A"], "Identity matrix (I)"),
    ],
    "NEET": [
        ("Universal Donor blood group is:", ["AB+", "O-", "A+", "B-"], "O-"),
        ("Functional unit of the kidney is:", ["Neuron", "Nephron", "Alveoli", "Villi"], "Nephron"),
        ("Scurvy is caused by deficiency of:", ["Vitamin A", "Vitamin B12", "Vitamin C", "Vitamin D"], "Vitamin C"),
        ("DNA replication occurs in which phase?", ["G1 phase", "S phase", "G2 phase", "M phase"], "S phase"),
        ("Normal human somatic cell has how many chromosomes?", ["23", "44", "46", "48"], "46"),
    ],
    "UPSC": [
        ("Preamble of Indian Constitution was amended by:", ["42nd Amendment (1976)", "44th Amendment", "73rd Amendment", "86th Amendment"], "42nd Amendment (1976)"),
        ("Which river is known as 'Dakshin Ganga'?", ["Krishna", "Kaveri", "Godavari", "Narmada"], "Godavari"),
        ("Monetary Policy in India is formulated by:", ["Finance Ministry", "Reserve Bank of India (RBI)", "SEBI", "NITI Aayog"], "Reserve Bank of India (RBI)"),
        ("Chairman of Drafting Committee was:", ["Dr. Rajendra Prasad", "Dr. B.R. Ambedkar", "Jawaharlal Nehru", "Sardar Patel"], "Dr. B.R. Ambedkar"),
        ("Anti-Defection Law is in which Schedule?", ["8th Schedule", "9th Schedule", "10th Schedule", "11th Schedule"], "10th Schedule"),
    ],
    "SSC & Govt Exams": [
        ("A 150m train crosses a pole in 15 sec. Speed in km/h:", ["36 km/h", "45 km/h", "54 km/h", "60 km/h"], "36 km/h"),
        ("Next number in series: 2, 6, 12, 20, 30, ?", ["40", "42", "44", "46"], "42"),
        ("SI on ₹5000 at 10% p.a. for 2 years is:", ["₹500", "₹1000", "₹1050", "₹1500"], "₹1000"),
        ("Who is known as 'Iron Man of India'?", ["Subhas Bose", "Sardar Vallabhbhai Patel", "Bhagat Singh", "Nehru"], "Sardar Vallabhbhai Patel"),
        ("Synonym of 'ABUNDANT' is:", ["Scarce", "Plentiful", "Empty", "Rare"], "Plentiful"),
    ],
    "Banking & Railway": [
        ("Regulator of the banking system in India is:", ["SEBI", "Reserve Bank of India (RBI)", "NABARD", "SIDBI"], "Reserve Bank of India (RBI)"),
        ("What does 'NEFT' stand for in Banking?", ["National Electronic Funds Transfer", "New Electronic Finance Transfer", "National Easy Fund Transfer", "None"], "National Electronic Funds Transfer"),
        ("Which mirror is used as a rear-view mirror in vehicles?", ["Concave mirror", "Convex mirror", "Plane mirror", "Cylindrical"], "Convex mirror"),
        ("Amount on ₹10,000 at 10% p.a. CI for 2 years:", ["₹12,000", "₹12,100", "₹11,000", "₹12,500"], "₹12,100"),
        ("If 15% of X is 45, what is X?", ["250", "300", "350", "400"], "300"),
    ],
    "SSB & Defence": [
        ("OIR: If 'ARMY' is 'BSNZ', how is 'NAVY' coded?", ["OBWZ", "OBXZ", "OCWZ", "MBWZ"], "OBWZ"),
        ("SRT: An elderly person falls on the road while you go for SSB. You:", ["Ignore and rush", "Help them up, arrange aid, then proceed", "Wait for others", "Leave immediately"], "Help them up, arrange aid, then proceed"),
        ("Which is NOT one of the 4 OLQ factors?", ["Planning & Organizing", "Social Adjustment", "Dynamic", "Financial Wealth"], "Financial Wealth"),
        ("National Defence Academy (NDA) is located at:", ["Dehradun", "Khadakwasla (Pune)", "Chennai", "Gaya"], "Khadakwasla (Pune)"),
        ("In SSB Screening, PPDT stands for:", ["Picture Perception & Description Test", "Personal Personality Test", "Physical Drill Test", "None"], "Picture Perception & Description Test"),
    ],
}

# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------
if "resources_df" not in st.session_state:
    st.session_state.resources_df = pd.DataFrame(INITIAL_DATA, columns=COLUMNS)
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_ai_explanation" not in st.session_state:
    st.session_state.last_ai_explanation = None
if "last_search_label" not in st.session_state:
    st.session_state.last_search_label = ""
if "bookmarks" not in st.session_state:
    st.session_state.bookmarks = []
if "completed_titles" not in st.session_state:
    st.session_state.completed_titles = set()
if "test_active" not in st.session_state:
    st.session_state.test_active = False
if "test_start_time" not in st.session_state:
    st.session_state.test_start_time = 0.0
if "test_duration_sec" not in st.session_state:
    st.session_state.test_duration_sec = 120
if "test_stream" not in st.session_state:
    st.session_state.test_stream = "Banking & Railway"
if "gemini_api_key" not in st.session_state:
    try:
        st.session_state.gemini_api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
if "working_model" not in st.session_state:
    st.session_state.working_model = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Namaste! Main **VidyaSpark AI Tutor (Powered by Google Gemini)** hoon. Apna koi bhi study doubt, coding error, numerical, ya exam question poochiye!"}
    ]

df = st.session_state.resources_df
STREAMS = ["All Streams"] + df["Stream"].drop_duplicates().tolist()
LEVELS = ["All Levels", "Beginner", "Intermediate", "Advanced"]
RESOURCE_TYPES = ["All Types"] + sorted(df["Resource_Type"].unique().tolist())


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def is_direct_yt_video(url):
    return "youtube.com/watch" in str(url) or "youtu.be/" in str(url)


def youtube_search_url(query_text, lang="All Languages"):
    lang_suffix = f" in {lang}" if lang != "All Languages" else ""
    return f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(f'{query_text}{lang_suffix}'.strip())}"


def recommend_resources(stream, topic, difficulty, language, resource_type, only_important=False, keywords="", limit=8):
    filtered = df.copy()
    lang_fallback_used = False
    if stream != "All Streams":
        filtered = filtered[filtered["Stream"] == stream]
    if topic != "All Topics":
        filtered = filtered[filtered["Topic"] == topic]
    if difficulty != "All Levels":
        f = filtered[filtered["Difficulty"] == difficulty]
        if not f.empty:
            filtered = f
    if resource_type != "All Types":
        f = filtered[filtered["Resource_Type"] == resource_type]
        if not f.empty:
            filtered = f
    if only_important:
        f = filtered[filtered["Important"] == True]
        if not f.empty:
            filtered = f
    if language != "All Languages":
        f = filtered[filtered["Language"] == language]
        if not f.empty:
            filtered = f
        else:
            lang_fallback_used = True

    if filtered.empty:
        return filtered, lang_fallback_used

    docs = (
        filtered["Stream"].fillna("") + " " + filtered["Topic"].fillna("") + " " +
        filtered["Title"].fillna("") + " " + filtered["Platform"].fillna("") + " " +
        filtered["Description"].fillna("")
    ).tolist()
    query = f"{stream} {topic} {difficulty} {language} {keywords}".strip()
    vec = TfidfVectorizer(stop_words="english")
    mat = vec.fit_transform(docs + [query])
    scores = cosine_similarity(mat[-1], mat[:-1]).flatten()
    filtered = filtered.copy()
    filtered["Similarity"] = scores
    return filtered.sort_values(by=["Important", "Similarity"], ascending=[False, False]).head(int(limit)), lang_fallback_used


def call_gemini(prompt, api_key):
    if not GENAI_AVAILABLE:
        return "⚠️ `google-genai` package install nahi hai. Terminal mein `pip install google-genai` chalaiye."
    if not api_key.strip():
        return "🔑 Left sidebar mein apni **Gemini API Key** paste kijiye taaki AI explanation aur doubt solver chal sake."

    try:
        client = genai.Client(api_key=api_key.strip())

        if st.session_state.working_model:
            try:
                resp = client.models.generate_content(model=st.session_state.working_model, contents=prompt)
                return resp.text
            except Exception:
                st.session_state.working_model = None

        candidates = ["gemini-2.0-flash", "gemini-flash-latest", "gemini-3-flash-preview"]
        for m in client.models.list():
            m_name = m.name.replace("models/", "")
            if "gemini" in m_name.lower() and "2.5-flash" not in m_name.lower() and m_name not in candidates:
                candidates.append(m_name)

        last_err = ""
        for model_name in candidates:
            try:
                resp = client.models.generate_content(model=model_name, contents=prompt)
                st.session_state.working_model = model_name
                return resp.text
            except Exception as e:
                last_err = str(e)
                continue

        return f"❌ Koi bhi Gemini model respond nahi kar raha. Last Error: `{last_err}`"
    except Exception as e:
        return f"❌ Gemini API Error: `{e}`"


def ask_gemini_tutor(user_question, api_key, stream_ctx, lang_pref, history):
    recent_chat = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history[-6:]])
    prompt = f"""You are VidyaSpark AI Tutor, a friendly, expert Indian teacher.
Student's Stream/Exam: {stream_ctx}
Preferred Language: {lang_pref}

Recent Conversation:
{recent_chat}

Current Student Question: {user_question}

Instructions:
- Answer clearly, accurately, and step-by-step in {lang_pref}.
- Use bullet points or formulas where helpful.
- Keep it exam-oriented and easy to understand for Indian students."""
    return call_gemini(prompt, api_key)


def explain_topic_with_gemini(topic_query, stream_ctx, difficulty_ctx, lang_pref, api_key):
    target_lang = "Hinglish" if lang_pref == "All Languages" else lang_pref
    prompt = f"""You are VidyaSpark, an expert Indian study mentor.
A student searched for the following study topic:
- Topic / Search Query: {topic_query}
- Stream / Exam: {stream_ctx}
- Level: {difficulty_ctx}
- Preferred Language: {target_lang}

Provide a crisp, student-friendly study guide in {target_lang} with:
1. **Simple Explanation** (what this topic is in 3-4 easy lines)
2. **Key Concepts / Formulas / Points** (4-5 bullet points)
3. **Quick Exam Tip or Example** (1 short practical tip/example so the student can learn right here)

Keep it well-structured, concise, and easy to read."""
    return call_gemini(prompt, api_key)


def get_difficulty_badge(level):
    if level == "Beginner":
        return '<span class="badge badge-beginner">🟢 Beginner</span>'
    elif level == "Intermediate":
        return '<span class="badge badge-intermediate">🟡 Intermediate</span>'
    return '<span class="badge badge-advanced">🔴 Advanced</span>'


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-brand"><h2>✨ VidyaSpark</h2><p>India\'s Smartest AI Study Companion</p></div>', unsafe_allow_html=True)

bm_count = len(st.session_state.bookmarks)
page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "🎯 Find Resources",
        "🤖 AI Doubt Solver (Gemini)",
        "📝 Timed Sample Papers",
        "🎥 YouTube Hub (All Lang)",
        f"⭐ My Study Plan ({bm_count})",
        "➕ Add Resource",
        "ℹ️ About Project",
    ],
)


st.sidebar.divider()
st.sidebar.markdown("**Developed by:**\n### Abhisek Kumar")
st.sidebar.caption("B.Tech CSE (AI & ML) | Reg: 125433")
st.sidebar.caption("10 Streams: CSE • Mech • Elec • Boards • JEE • NEET • UPSC • SSC • Banking/Railway • SSB")


# ---------------------------------------------------------
# PAGE 1: DASHBOARD
# ---------------------------------------------------------
if page == "📊 Dashboard":
    st.markdown(
        '<div class="hero-banner"><h1>✨ Welcome to VidyaSpark</h1>'
        '<p>10 Streams (incl. Banking & Railway) • Live Gemini AI Explanations & Doubt Solver • Timed Sample Papers.</p></div>',
        unsafe_allow_html=True,
    )

    imp_count = int(df["Important"].sum())
    done, tot = len(st.session_state.completed_titles), len(st.session_state.bookmarks)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="kpi-card kpi-blue"><div class="kpi-value">{len(df)}</div><div class="kpi-label">📚 Curated Resources</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-card kpi-orange"><div class="kpi-value">{imp_count}</div><div class="kpi-label">🔥 Important Topics</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-card kpi-purple"><div class="kpi-value">{len(SAMPLE_PAPERS)}</div><div class="kpi-label">📝 Timed Sample Papers</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi-card kpi-green"><div class="kpi-value">{done}/{tot}</div><div class="kpi-label">✅ Study Plan Done</div></div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("🔥 Must-Study Important Topics")
    dash_stream = st.selectbox("Filter by Stream", STREAMS, key="dash_imp_stream")
    imp_df = df[df["Important"] == True]
    if dash_stream != "All Streams":
        imp_df = imp_df[imp_df["Stream"] == dash_stream]

    imp_cols = st.columns(2)
    for idx, (_, row) in enumerate(imp_df.head(6).iterrows()):
        with imp_cols[idx % 2]:
            st.markdown(
                f'<div class="imp-box"><strong style="color:#FDE047;">🔥 {row["Topic"]}</strong> '
                f'<span style="color:#38BDF8; font-weight:600;">• {row["Stream"]}</span><br/>'
                f'<span style="font-size:0.86rem; color:#E2E8F0;">{row["Title"]} ({row["Platform"]} • {row["Language"]})</span></div>',
                unsafe_allow_html=True,
            )

    st.divider()
    ch1, ch2 = st.columns(2)
    with ch1:
        st.subheader("🎓 Resources by Stream / Exam")
        st.bar_chart(df["Stream"].value_counts(), color="#38BDF8")
    with ch2:
        st.subheader("🗂️ Resources by Type")
        st.bar_chart(df["Resource_Type"].value_counts(), color="#EC4899")


# ---------------------------------------------------------
# PAGE 2: FIND RESOURCES (WITH LIVE GEMINI EXPLANATION)
# ---------------------------------------------------------
elif page == "🎯 Find Resources":
    st.markdown('<div class="hero-title">🎯 Find Best Study Material</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Curated resources + Instant Google Gemini AI Explanation for any topic you search!</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    stream = c1.selectbox("1. Stream / Exam", STREAMS)
    avail_topics = ["All Topics"] + sorted((df if stream == "All Streams" else df[df["Stream"] == stream])["Topic"].unique().tolist())
    topic = c2.selectbox("2. Topic", avail_topics)
    language = c3.selectbox("3. Language", ALL_LANGUAGES)
    difficulty = c4.selectbox("4. Difficulty", LEVELS)

    c5, c6, c7 = st.columns([2.5, 1.2, 1.3])
    keywords = c5.text_input("Search Any Topic / Teacher / Concept", placeholder="e.g., Thermodynamics, Python Loops, Repo Rate, Integration")
    resource_type = c6.selectbox("Resource Type", RESOURCE_TYPES)
    with c7:
        st.write("")
        st.write("")
        only_important = st.toggle("🔥 Only Important", value=False)

    b1, b2 = st.columns([3, 1])
    search_btn = b1.button("🚀 Get AI Recommendations & Explanation", type="primary", use_container_width=True)
    yt_q = f"{stream if stream != 'All Streams' else ''} {topic if topic != 'All Topics' else ''} {keywords} lecture".strip()
    b2.link_button(f"🎥 YouTube ({language}) ↗", youtube_search_url(yt_q, language), use_container_width=True)

    if search_btn:
        st.session_state.last_results = recommend_resources(stream, topic, difficulty, language, resource_type, only_important, keywords) + (language,)

        # Build clear topic label for Gemini explanation
        search_label = keywords.strip() if keywords.strip() else (topic if topic != "All Topics" else (stream if stream != "All Streams" else "Effective Study Strategy"))
        st.session_state.last_search_label = search_label

        if st.session_state.gemini_api_key.strip():
            with st.spinner(f"✨ Gemini '{search_label}' ki explanation tayyar kar raha hai..."):
                st.session_state.last_ai_explanation = explain_topic_with_gemini(
                    search_label, stream, difficulty, language, st.session_state.gemini_api_key
                )
        else:
            st.session_state.last_ai_explanation = "NO_KEY"

    # Show Gemini Topic Explanation Card
    if st.session_state.last_ai_explanation:
        st.divider()
        st.markdown(f"### ✨ Gemini AI Quick Study Guide: `{st.session_state.last_search_label}`")
        if st.session_state.last_ai_explanation == "NO_KEY":
            st.info("🔑 Left sidebar mein apni **Gemini API Key** paste karein taaki har search ke saath yahan topic ki instant AI explanation bhi mile!")
        else:
            with st.container(border=True):
                st.markdown(st.session_state.last_ai_explanation)

    # Show Recommended Study Materials
    if st.session_state.last_results:
        results, fb, sel_lang = st.session_state.last_results
        st.divider()
        st.markdown("### 📚 Recommended Study Resources")
        if fb:
            st.info(f"Showing best curated resources. Click **🎥 YouTube ({sel_lang}) ↗** above for instant lectures in **{sel_lang}**!")
        for idx, item in results.iterrows():
            with st.container(border=True):
                imp_tag = '<span class="badge badge-imp">🔥 IMPORTANT</span>' if item["Important"] else ""
                st.markdown(f"### 📘 {item['Title']}")
                st.markdown(
                    imp_tag + f'<span class="badge badge-meta">🎓 {item["Stream"]} • {item["Topic"]}</span>'
                    + get_difficulty_badge(item["Difficulty"]) + f'<span class="badge badge-lang">🗣️ {item["Language"]}</span>',
                    unsafe_allow_html=True,
                )
                st.write(item["Description"])
                if is_direct_yt_video(item["Link"]):
                    with st.expander("▶️ Watch Video Inside App"):
                        st.video(item["Link"])
                btn1, btn2 = st.columns(2)
                btn1.link_button("Open Resource ↗", item["Link"], use_container_width=True)
                if any(b["Title"] == item["Title"] for b in st.session_state.bookmarks):
                    btn2.button("✅ Saved", key=f"sv_{idx}", disabled=True, use_container_width=True)
                elif btn2.button("⭐ Save to Study Plan", key=f"add_{idx}", use_container_width=True):
                    st.session_state.bookmarks.append(item.drop("Similarity").to_dict())
                    st.rerun()


# ---------------------------------------------------------
# PAGE 3: AI DOUBT SOLVER (POWERED BY GOOGLE GEMINI)
# ---------------------------------------------------------
elif page == "🤖 AI Doubt Solver (Gemini)":
    st.markdown('<div class="hero-title">🤖 VidyaSpark AI Doubt Solver</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Powered by Google Gemini — koi bhi question, numerical, coding bug ya exam doubt seedha poochiye!</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 2, 1])
    ai_stream = c1.selectbox("Select Your Stream / Exam", STREAMS[1:], key="ai_st")
    ai_lang = c2.selectbox("Answer Language", ["Hinglish", "English", "Hindi"], key="ai_lg")
    with c3:
        st.write("")
        st.write("")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = [st.session_state.chat_history[0]]
            st.rerun()

    if not st.session_state.gemini_api_key.strip():
        st.warning("🔑 Left sidebar mein apni **Gemini API Key** paste karein to activate live AI answers. (Free key: https://aistudio.google.com/apikey)")

    st.divider()
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Apna doubt ya question yahan likhiye (e.g., Python sikhao / Repo Rate kya hai)")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("✨ Gemini soch raha hai..."):
                ai_reply = ask_gemini_tutor(
                    user_q,
                    st.session_state.gemini_api_key,
                    ai_stream,
                    ai_lang,
                    st.session_state.chat_history[:-1],
                )
            st.markdown(ai_reply)

        st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})


# ---------------------------------------------------------
# PAGE 4: TIMED SAMPLE PAPERS (ALL 10 STREAMS)
# ---------------------------------------------------------
elif page == "📝 Timed Sample Papers":
    st.markdown('<div class="hero-title">📝 Timed Sample Papers & Mock Tests</div>', unsafe_allow_html=True)

    if not st.session_state.test_active:
        c1, c2 = st.columns(2)
        selected_paper = c1.selectbox("Select Exam / Stream", list(SAMPLE_PAPERS.keys()))
        time_choice = c2.selectbox("Select Time Limit", ["2 Minutes (Rapid)", "5 Minutes (Standard)", "10 Minutes (Relaxed)"])
        dur_map = {"2 Minutes (Rapid)": 120, "5 Minutes (Standard)": 300, "10 Minutes (Relaxed)": 600}

        if st.button("⏱️ Start Timed Sample Paper", type="primary", use_container_width=True):
            st.session_state.test_active = True
            st.session_state.test_stream = selected_paper
            st.session_state.test_duration_sec = dur_map[time_choice]
            st.session_state.test_start_time = time.time()
            st.rerun()
    else:
        paper_name = st.session_state.test_stream
        total_sec = st.session_state.test_duration_sec
        remaining_sec = max(0, total_sec - int(time.time() - st.session_state.test_start_time))

        components.html(
            f"""
            <div style="background: linear-gradient(90deg, #DC2626, #EA580C); color: white; padding: 12px 18px; border-radius: 12px; font-family: sans-serif; display: flex; justify-content: space-between; align-items: center;">
                <div style="font-size: 1.05rem; font-weight: 700;">📝 {paper_name} — Live Sample Paper</div>
                <div id="timer" style="font-size: 1.35rem; font-weight: 800; background: rgba(0,0,0,0.25); padding: 5px 12px; border-radius: 8px;">⏱️</div>
            </div>
            <script>
                let t = {remaining_sec};
                const el = document.getElementById('timer');
                function tick() {{
                    if (t <= 0) {{ el.innerHTML = "⏰ TIME UP!"; return; }}
                    el.innerHTML = "⏱️ " + String(Math.floor(t/60)).padStart(2,'0') + ":" + String(t%60).padStart(2,'0');
                    t--;
                }}
                tick(); setInterval(tick, 1000);
            </script>
            """,
            height=70,
        )

        questions = SAMPLE_PAPERS[paper_name]
        answers = {i: st.radio(f"**Q{i+1}. {q}**", opts, key=f"sp_{paper_name}_{i}") for i, (q, opts, _) in enumerate(questions)}

        c_sub, c_exit = st.columns([3, 1])
        submit_paper = c_sub.button("✅ Submit Sample Paper", type="primary", use_container_width=True)
        if c_exit.button("❌ Exit", use_container_width=True):
            st.session_state.test_active = False
            st.rerun()

        if submit_paper:
            taken = int(time.time() - st.session_state.test_start_time)
            st.session_state.test_active = False
            rows = [{"Q.No": f"Q{i+1}", "Question": q, "Your Answer": answers[i], "Correct": ans, "Status": "✅" if answers[i] == ans else "❌"} for i, (q, _, ans) in enumerate(questions)]
            score = sum(1 for r in rows if r["Status"] == "✅")
            st.success(f"🎉 Score: **{score}/{len(questions)}** | Time Taken: **{taken} sec** (Limit: {total_sec} sec)")
            rep_df = pd.DataFrame(rows)
            st.dataframe(rep_df, use_container_width=True)
            st.download_button("⬇️ Download Report (CSV)", rep_df.to_csv(index=False).encode("utf-8"), file_name=f"{paper_name}_report.csv")


# ---------------------------------------------------------
# PAGE 5: YOUTUBE HUB
# ---------------------------------------------------------
elif page == "🎥 YouTube Hub (All Lang)":
    st.markdown('<div class="hero-title">🎥 YouTube Learning Hub (All Languages)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    yt_stream = c1.selectbox("Select Stream", STREAMS, key="yt_st")
    yt_lang = c2.selectbox("Video Language", ALL_LANGUAGES, key="yt_lg")
    custom_q = st.text_input("Search Any Topic or Teacher on YouTube", placeholder="e.g., Sahil Sir Railway Maths, PW Physics, Khan Sir GS")
    if custom_q.strip():
        st.link_button(f"🔍 Watch '{custom_q.strip()}' ({yt_lang}) on YouTube ↗", youtube_search_url(custom_q.strip(), yt_lang), type="primary")

    st.divider()
    yt_df = df[df["Link"].str.contains("youtube.com", na=False)]
    if yt_stream != "All Streams":
        yt_df = yt_df[yt_df["Stream"] == yt_stream]
    cols = st.columns(2)
    for i, (_, row) in enumerate(yt_df.iterrows()):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"#### 🎬 {row['Title']}")
                st.caption(f"{row['Stream']} • {row['Language']} • {row['Platform']}")
                if is_direct_yt_video(row["Link"]):
                    st.video(row["Link"])
                else:
                    st.link_button("▶️ Open on YouTube ↗", row["Link"], use_container_width=True)


# ---------------------------------------------------------
# PAGE 6: MY STUDY PLAN
# ---------------------------------------------------------
elif page.startswith("⭐ My Study Plan"):
    st.markdown('<div class="hero-title">⭐ My Saved Study Plan</div>', unsafe_allow_html=True)
    if not st.session_state.bookmarks:
        st.info("Your Study Plan is empty. Save resources from **Find Resources**!")
    else:
        st.download_button("⬇️ Export Plan (CSV)", pd.DataFrame(st.session_state.bookmarks).to_csv(index=False).encode("utf-8"), file_name="study_plan.csv")
        for i, item in enumerate(st.session_state.bookmarks):
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                done = item["Title"] in st.session_state.completed_titles
                if c1.checkbox(f"**{item['Title']}** ({item['Stream']})", value=done, key=f"dn_{i}") != done:
                    (st.session_state.completed_titles.discard if done else st.session_state.completed_titles.add)(item["Title"])
                    st.rerun()
                c2.link_button("Open ↗", item["Link"], use_container_width=True)


# ---------------------------------------------------------
# PAGE 7: ADD RESOURCE
# ---------------------------------------------------------
elif page == "➕ Add Resource":
    st.markdown('<div class="hero-title">➕ Contribute a Study Resource</div>', unsafe_allow_html=True)
    with st.form("add_res_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns(4)
        n_stream = c1.selectbox("Stream", df["Stream"].drop_duplicates().tolist())
        n_diff = c2.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced"])
        n_lang = c3.selectbox("Language", ALL_LANGUAGES[1:])
        n_type = c4.selectbox("Type", ["YouTube Video", "App / Course", "Course", "Documentation", "Practice"])
        n_topic = st.text_input("Topic")
        n_title = st.text_input("Resource Title")
        n_link = st.text_input("URL / YouTube Link")
        if st.form_submit_button("➕ Add to Platform", type="primary") and n_title.strip() and n_link.strip():
            new_row = pd.DataFrame([[n_stream, n_topic or n_stream, n_diff, n_type, n_lang, "Community", "Self-paced", True, n_title.strip(), "User contributed resource.", n_link.strip()]], columns=COLUMNS)
            st.session_state.resources_df = pd.concat([st.session_state.resources_df, new_row], ignore_index=True)
            st.success(f"✅ **{n_title}** added!")


# ---------------------------------------------------------
# PAGE 8: ABOUT PROJECT
# ---------------------------------------------------------
elif page == "ℹ️ About Project":
    st.markdown('<div class="hero-title">ℹ️ About VidyaSpark</div>', unsafe_allow_html=True)
    st.write("**VidyaSpark** — All-India AI Study Platform covering 10 Streams (including Banking & Railway) with live **Google Gemini Topic Explanations**, AI Doubt Solver, Timed Sample Papers, and All-Language YouTube Hub.")
    st.write("**Developer:** Abhisek Kumar (Reg: 125433) | B.Tech CSE (AI & ML)")
