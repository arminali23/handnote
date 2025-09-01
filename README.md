# HandNote Web

**HandNote** is a web-based AI-powered notepad that lets you draw letters on a digital canvas or in the air with your hand (via webcam + pinch gesture).  
The app recognizes your characters using a CNN trained on the EMNIST dataset, builds your text, and allows you to save notes.

---

### Features
-  **Canvas Drawing**: Write letters with mouse/touch, then predict with one click.  
-  **Webcam + Pinch**: Draw in the air (thumb-index pinch). Toggle webcam mode.  
-  **Character Recognition**: CNN trained on EMNIST Letters (A–Z).  
-  **Digits & Punctuation**: Add numbers (0–9) and punctuation (`. , ! ? : ;`) via keyboard.  
-  **Text Buffer**: Collects characters into a live note.  
-  **Top-3 Predictions**: Quickly replace last char with keys `1/2/3`.  
-  **Save Notes**: Export notes as `.txt` files.  
-  **Undo / Backspace / Enter / Space** controls.  

---

###  Project Structure

webapp/
server/
app.py                  # FastAPI backend
requirements.txt
emnist_letters_cnn.pt   # trained CNN model
notes/                  # saved text notes
client/
index.html              # Web frontend


---

###  Tech Stack
- **Frontend**: HTML5 Canvas, JavaScript, MediaPipe Hands  
- **Backend**: Python 3.10+, FastAPI, PyTorch, OpenCV, NumPy  

---

###  Setup & Run

#### Clone
```bash
git clone https://github.com/yourusername/handnote.git
cd handnote/webapp
### Backend
cd server
python -m venv venv
source venv/bin/activate   # (Windows: venv\Scripts\activate)
pip install -r requirements.txt

# copy your trained model here:
# emnist_letters_cnn.pt
uvicorn app:app --reload --port 8000

### Frontend

cd ../client
python -m http.server 5500


Usage
	1.	Draw one letter on the canvas (mouse/touch) or in the air (webcam + pinch).
	2.	Click Predict → recognized character appears in text buffer.
	3.	If wrong, check Top-3 and press 1/2/3 to correct.
	4.	Add spaces, newlines, backspace, undo.
	5.	Save note as .txt file in server/notes/.

⸻

Model Details
	•	Dataset: EMNIST Letters (26 classes A–Z)
	•	Architecture: 2 Conv blocks → Dense → Dropout → Output
	•	Training accuracy ~94%, Validation accuracy ~94%
	•	Preprocessing: deskew, resize, center-of-mass normalization, TTA (rotations, erosion/dilation)
