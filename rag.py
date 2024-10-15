from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# Import the Python SDK
import google.generativeai as genai

app = Flask(__name__)
CORS(app)  # Enable CORS
socketio = SocketIO(app, cors_allowed_origins='*')

review_df = pd.read_csv("inputdata.csv", index_col=[0])
review_df["Clothing ID"].value_counts()

#Group the reviews_df DataFrame by 'Clothing ID'
grouped_reviews_df = review_df.groupby('Clothing ID')

# Collect all groups into a list
group_list = [group for _, group in grouped_reviews_df]

# Concatenate all groups into a single DataFrame
reviews_df = pd.concat(group_list)

# Display the concatenated DataFrame
print(reviews_df)

def text_concat_info(row):
    _id = row["Clothing ID"]
    _title = row["Title"]
    _review_text = row["Review Text"]
    _class_name = row["Class Name"]
    return 'Product with ID: ' + str(_id) + ' and title: ' + str(_title) + ' has review: ' + str(_review_text) + ' and class name: ' + str(_class_name)

reviews_df["Result Text"] = reviews_df.apply(text_concat_info, axis=1)

reviews_df = reviews_df.dropna()

def text_preprocessing(text: str):
    text = text.replace("\n", " ")
    text = text.replace("  ", " ")
    return text

reviews_df = reviews_df.drop_duplicates(subset=["Result Text"])

text_splitter = RecursiveCharacterTextSplitter(
    separators = ["."],
    chunk_size = 400,
    chunk_overlap  = 0,
    is_separator_regex = False,
)

def split_into_chunks(text):
    docs = text_splitter.create_documents([text])
    text_chunks = [doc.page_content for doc in docs]

    return text_chunks

reviews_df["text_chunk"] = reviews_df["Result Text"].apply(split_into_chunks)
reviews_df = reviews_df.explode("text_chunk")
reviews_df["chunk_id"] = reviews_df.groupby(level=0).cumcount()

# Create Document Embeddings Vectors
model_name = "all-mpnet-base-v2"
model = SentenceTransformer(model_name)

text_chunks = reviews_df["text_chunk"].tolist()
text_chunk_vectors = model.encode(text_chunks, show_progress_bar=True)

def retrieve_relevant_documents(query, text_chunk_vectors, k):
    query_embedding = model.encode(query)

    # Calculate cosine similarity between the query and each document
    similarities = cosine_similarity([query_embedding], text_chunk_vectors)[0]

    # Get indices of the top k most similar documents
    top_k_indices = np.argsort(similarities)[::-1][:k]

    # Retrieve the relevant documents
    return reviews_df.iloc[top_k_indices]

relevant_rows = retrieve_relevant_documents("Fabric quality", text_chunk_vectors, 5)

prompt_template = """
Bạn là một trợ lý trò chuyện hữu ích tên là MaiDora. Tôi sẽ cung cấp cho bạn một số đánh giá sản phẩm trên một nền tảng thương mại điện tử về một sản phẩm cụ thể.
Tôi muốn bạn xem xét các ý kiến khác nhau trong thông tin mà tôi cung cấp và tạo ra một câu trả lời ngắn gọn, chính xác và đủ thông tin bằng tiếng Việt cho câu hỏi của người dùng.
Hãy nói "Tôi không biết" nếu bạn được hỏi một câu hỏi mà bạn không biết câu trả lời, đừng tự nghĩ ra câu trả lời.
Nếu bạn muốn đề xuất một sản phẩm, vui lòng viết mã sản phẩm và tiêu đề dưới dạng: ([PRODUCT_ID][PRODUCT_TITLE]).
Nếu không thể đề xuất một sản phẩm phù hợp với câu hỏi của người dùng, hãy đề xuất một sản phẩm được đánh giá cao.

Dưới đây là các tài liệu:
<documents>

Câu hỏi của người dùng: <query>
"""

def create_prompt(query, k=5):
    # get relevant information about the query, and append this information into the prompt template
    # you can get any information you like.
    relevant_rows = retrieve_relevant_documents(query, text_chunk_vectors, k)
    text_chunks = relevant_rows["text_chunk"].tolist()
    text_chunks_string = "\n".join(text_chunks)

    prompt = prompt_template
    prompt = prompt.replace("<documents>", text_chunks_string)
    prompt = prompt.replace("<query>", query)

    return prompt

prompt = create_prompt("Fabric Quality")
print(prompt)

# Answering the user query
genai.configure(api_key='Gemini_KEY')

geminiModel = genai.GenerativeModel('gemini-1.5-flash')


def generate(prompt):
    response = geminiModel.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            max_output_tokens=256,
            temperature=0.2
        )
    )

    return response.text

user_query = "I'am 5'2 and 130 lbs.What size should I get?"
prompt = create_prompt(user_query)
answer = generate(prompt)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('message')
def handle_message(data):
    print('Received message:', data)
    user_query = data.get('query')
    if user_query:
        prompt = create_prompt(user_query)
        answer = generate(prompt)
        emit('response', {'answer': answer})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=4646)