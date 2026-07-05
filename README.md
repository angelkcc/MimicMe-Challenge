# 🎭 MimicMe CHALLENGE
### A motion-based meme gesture game for college exhibitions

---

## 📦 SETUP (One-Time)

Open your terminal / VS Code terminal and run:

```bash
pip install -r requirements.txt
```

> ⚠️  Make sure you install **mediapipe==0.10.14** exactly (not a newer version).
> Newer versions of mediapipe removed the `solutions` API used by this game.

---

## ▶️ HOW TO RUN

```bash
python meme_game.py
```

Make sure your **webcam** is connected and allowed.

---

## 🕹️ HOW TO PLAY

| Step | Action |
|------|--------|
| 1 | A meme image appears on the top-right of the screen |
| 2 | Read the instruction at the bottom |
| 3 | Strike the pose with your body/hands |
| 4 | The accuracy bar fills up as you match the pose |
| 5 | **Hold the pose** for ~0.6 seconds to advance |
| 6 | After all 6 memes → Final Score Screen! |

---

## 🎭 THE 6 MEMES & POSES

| # | Meme | Gesture to do |
|---|------|---------------|
| 1 | Monkey Thinking | Raise one hand up near your chin/cheek |
| 2 | Monkey Shocked | Clasp both hands together at chest level |
| 3 | Monkey Screaming | Raise both hands up to the sides of your head |
| 4 | Finger Guns | Point both index fingers forward (finger guns!) |
| 5 | Sneaky Cat | Spread both arms LOW and wide |
| 6 | Shocked Cat | Put both hands on your cheeks (Home Alone pose!) |

---

## 💡 TIPS FOR EXHIBITION

- Stand ~1–1.5 metres from the camera
- Make sure your upper body (shoulders + hands) are visible
- Good lighting helps the pose detection
- The screen mirrors your movement so it feels natural

---

## 📁 FILE STRUCTURE

```
hex_meme_game/
├── meme_game.py          ← Main game file
├── requirements.txt      ← Python dependencies
├── README.md             ← This file
└── memes/
    ├── meme1_monkey_thinking.jpg
    ├── meme2_monkey_shocked.jpg
    ├── meme3_monkey_screaming.jpg
    ├── meme4_monkey_fingerguns.jpg
    ├── meme5_cat_sneaky.jpg
    └── meme6_cat_shocked.jpg
```

---

## 🔧 DEPENDENCIES

| Package | Version | Why |
|---------|---------|-----|
| `opencv-python` | ≥ 4.8 | Webcam capture + drawing |
| `mediapipe` | **== 0.10.14** | Hand & pose landmark detection |
| `numpy` | ≥ 1.24 | Math / array operations |
| `Pillow` | ≥ 9.0 | Image utilities |

---

## ❓ TROUBLESHOOTING

**Camera not opening?**
→ Make sure no other app is using your webcam. Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`.

**Import errors?**
→ Run `pip install -r requirements.txt` again.

**Poses not detecting?**
→ Ensure you're well-lit and your upper body is fully in frame.


