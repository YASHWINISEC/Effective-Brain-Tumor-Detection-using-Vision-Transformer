from flask import Flask, render_template, request, send_file
import torch, os, cv2
from cnn_model import SimpleCNN
from predict import analyze_image
from report import generate_report

app = Flask(__name__)

UPLOAD = "static/uploads"
OUTPUT = "static/outputs"
REPORTS = "static/reports"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)

DEVICE = torch.device("cpu")

model = SimpleCNN()
model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
model.eval()

print("✅ Model loaded correctly")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    path = os.path.join(UPLOAD, file.filename)
    file.save(path)

    label, conf, cam, box = analyze_image(model, path, DEVICE)

    img = cv2.imread(path)
    img = cv2.resize(img, (224,224))

    heat = cv2.applyColorMap((cam*255).astype("uint8"), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heat, 0.4, 0)

    if box:
        x,y,w,h = box
        cv2.rectangle(overlay, (x,y), (x+w,y+h), (0,0,255), 2)

    out_path = os.path.join(OUTPUT, file.filename)
    cv2.imwrite(out_path, overlay)

    report_path = os.path.join(REPORTS, "report.pdf")
    generate_report(report_path, label, f"{conf*100:.2f}")

    return render_template(
        "index.html",
        prediction=label,
        confidence=f"{conf*100:.2f}",
        original=path,
        gradcam=out_path
    )

@app.route("/report")
def report():
    return send_file("static/reports/report.pdf", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
