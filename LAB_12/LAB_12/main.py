from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def tf_idf_retrieval(documents: list[str], query: str) -> pd.DataFrame:
    # Twoje rozwiązanie
    pass

documents = [
    "AI in healthcare improves diagnostics and patient care. AI algorithms analyze medical medical datasets efficiently",
    "AI enhances the e-commerce experience. AI systems recommend products, predict customer preferences, and increase sales. AI also personalizes online shopping",
    "Healthcare and AI are transforming diagnostics. Healthcare diagnostics benefit greatly from AI’s ability to process large datasets efficiently",
    "AI applications in healthcare include diagnostics, treatment planning, and patient monitoring. Healthcare systems increasingly rely on AI advancements",
    "AI in transportation is reshaping the future with self-driving cars, traffic optimization, and route planning. AI is making travel safer and more efficient"
]

query = "AI in healthcare diagnostics"
results = tf_idf_retrieval(documents, query)
print(results)