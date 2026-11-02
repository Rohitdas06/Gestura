import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

PDF_FILENAME = "Gestura_Technical_Architecture.pdf"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (Pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "Gestura — Technical Architecture & System Specification")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)

        # Footer (All pages)
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 48, 8.5 * inch - 54, 48)
        
        self.drawString(54, 34, "CONFIDENTIAL & PROPRIETARY — GESTURA AI SYSTEMS")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 34, page_str)
        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        PDF_FILENAME,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Palettes
    PRIMARY = colors.HexColor("#0F172A")    # Slate 900
    ACCENT = colors.HexColor("#2563EB")     # Blue 600
    TEXT_COLOR = colors.HexColor("#334155") # Slate 700
    LIGHT_BG = colors.HexColor("#F8FAFC")   # Slate 50
    BORDER_COLOR = colors.HexColor("#E2E8F0")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=PRIMARY,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=ACCENT
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=ACCENT,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=TEXT_COLOR,
        spaceAfter=6
    )

    body_bold = ParagraphStyle(
        'Body_Bold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=TEXT_COLOR
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell,
        fontName='Helvetica-Bold',
        textColor=PRIMARY
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E293B")
    )

    elements = []

    # -------------------------------------------------------------
    # DOCUMENT COVER HEADER
    # -------------------------------------------------------------
    elements.append(Paragraph("GESTURA: Mobile App Architecture Specification", title_style))
    elements.append(Paragraph("Real-Time On-Device Sign Language Gesture Recognition & Speech Translation System", subtitle_style))
    elements.append(Paragraph("TECHNICAL WHITE PAPER & COMPREHENSIVE REPOSITORY SPECIFICATION &bull; GESTURA v1.0", meta_style))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT, spaceBefore=0, spaceAfter=14))

    # Executive Summary Callout Box
    summary_text = (
        "<b>Executive Summary:</b> Gestura is a high-performance, real-time sign language interpreter built on React Native "
        "and Expo. It combines hardware-accelerated video capture (VisionCamera v5), Google MediaPipe 21-point hand tracking "
        "(via a custom native Android Kotlin module), on-device neural network classification (Fast-TFLite v3), and native "
        "text-to-speech synthesis (Expo-Speech). This document provides an exhaustive technical analysis of the application's "
        "architecture, repository structure, end-to-end data pipelines, and core source modules."
    )
    summary_table = Table(
        [[Paragraph(summary_text, callout_style)]],
        colWidths=[7.0 * inch]
    )
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_BG),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('LINELEFT', (0, 0), (0, -1), 3.0, ACCENT),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 14))

    # -------------------------------------------------------------
    # SECTION 1: CORE TECH STACK & LIBRARIES
    # -------------------------------------------------------------
    elements.append(Paragraph("1. Core Technology Stack & Dependencies", h1_style))
    elements.append(Paragraph(
        "The application architecture relies on five specialized subsystems designed to achieve sub-50ms inference latency "
        "without triggering garbage collection overhead or camera pipeline locks.",
        body_style
    ))

    stack_data = [
        [
            Paragraph("Subsystem", table_header),
            Paragraph("Core Library / Engine", table_header),
            Paragraph("Version", table_header),
            Paragraph("Architectural Role & Significance", table_header)
        ],
        [
            Paragraph("Core Runtime", table_cell_bold),
            Paragraph("React Native & Expo SDK", table_cell),
            Paragraph("RN 0.86.3 / SDK 57", table_cell),
            Paragraph("React 19 New Architecture, Hermes JS engine, Expo CLI, and modular Nitro bindings.", table_cell)
        ],
        [
            Paragraph("Application Routing", table_cell_bold),
            Paragraph("Expo Router", table_cell),
            Paragraph("~57.0.29", table_cell),
            Paragraph("File-based navigation (app/index.tsx, app/explore.tsx) with native tab layout caching.", table_cell)
        ],
        [
            Paragraph("Camera Pipeline", table_cell_bold),
            Paragraph("react-native-vision-camera", table_cell),
            Paragraph("^5.2.3 (Nitro)", table_cell),
            Paragraph("Low-level Android CameraX/Camera2 wrapper providing multi-output preview streaming and still capture.", table_cell)
        ],
        [
            Paragraph("Vision / Tracking", table_cell_bold),
            Paragraph("Google MediaPipe Tasks Vision", table_cell),
            Paragraph("0.10.18", table_cell),
            Paragraph("Extracts 21 3D hand joints (63 spatial floats) via hand_landmarker.task running on device CPU/GPU.", table_cell)
        ],
        [
            Paragraph("Native Bridge", table_cell_bold),
            Paragraph("expo-mediapipe-hands", table_cell),
            Paragraph("Local Module", table_cell),
            Paragraph("Custom Kotlin Expo module handling EXIF rotation, adaptive downsampling, and JNI bitmap recycling.", table_cell)
        ],
        [
            Paragraph("Neural Inference", table_cell_bold),
            Paragraph("react-native-fast-tflite", table_cell),
            Paragraph("^3.0.1", table_cell),
            Paragraph("C++ LiteRT/TensorFlow Lite runtime with zero-copy ArrayBuffer bindings running models in <1 ms.", table_cell)
        ],
        [
            Paragraph("Speech Engine", table_cell_bold),
            Paragraph("expo-speech", table_cell),
            Paragraph("~57.0.3", table_cell),
            Paragraph("Native Android Text-To-Speech (TTS) synthesizer for audible word feedback with speech debouncing.", table_cell)
        ],
    ]

    t_stack = Table(stack_data, colWidths=[1.3 * inch, 1.8 * inch, 1.0 * inch, 2.9 * inch])
    t_stack.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
    ]))
    elements.append(t_stack)
    elements.append(Spacer(1, 14))

    # -------------------------------------------------------------
    # SECTION 2: REPOSITORY & DIRECTORY STRUCTURE
    # -------------------------------------------------------------
    elements.append(Paragraph("2. Repository Structure & Key Subsystems", h1_style))
    elements.append(Paragraph(
        "The project is structured into three distinct domains: mobile application runtime (GesturaMobile), "
        "machine learning data & training pipelines (ml_pipeline), and desktop research/validation tools.",
        body_style
    ))

    repo_data = [
        [
            Paragraph("Directory / Path", table_header),
            Paragraph("Subsystem Scope", table_header),
            Paragraph("Core Files & Responsibilities", table_header)
        ],
        [
            Paragraph("GesturaMobile/src/app/", table_cell_bold),
            Paragraph("Navigation Routes", table_cell),
            Paragraph("<b>index.tsx:</b> Active translation viewfinder with HUD, skeleton, & controls.<br/><b>explore.tsx:</b> Searchable sign dictionary with placement tips.<br/><b>_layout.tsx:</b> Tab bar navigation structure.", table_cell)
        ],
        [
            Paragraph("GesturaMobile/modules/expo-mediapipe-hands/", table_cell_bold),
            Paragraph("Custom Native Module", table_cell),
            Paragraph("<b>android/.../ExpoMediapipeHandsModule.kt:</b> Kotlin bridge invoking MediaPipe HandLandmarker with bitmap downsampling and EXIF rotation.<br/><b>src/index.ts:</b> TypeScript typing & safe fallback.", table_cell)
        ],
        [
            Paragraph("GesturaMobile/assets/model/", table_cell_bold),
            Paragraph("On-Device AI Models", table_cell),
            Paragraph("<b>sign_language_model.tflite:</b> 8-class standard gesture classifier (63 inputs).<br/><b>gesture_model_indian.tflite:</b> 35-class Indian Sign Language alphabet/number model.<br/><b>*.json:</b> Label mappings for all models.", table_cell)
        ],
        [
            Paragraph("GesturaMobile/src/lib/", table_cell_bold),
            Paragraph("Feature Computation", table_cell),
            Paragraph("<b>handDetector.ts:</b> Bridge invoking native module & wrist-centering normalizer.<br/><b>features.ts:</b> 132-d spatial vector calculations.<br/><b>signEngine.ts:</b> Sliding window temporal consensus engine.", table_cell)
        ],
        [
            Paragraph("GesturaMobile/src/components/", table_cell_bold),
            Paragraph("Visual Overlays", table_cell),
            Paragraph("<b>HandSkeletonOverlay.tsx:</b> Lightweight vector canvas rendering 21 hand joints, connecting bones, and dynamic target framing reticle.", table_cell)
        ],
        [
            Paragraph("ml_pipeline/", table_cell_bold),
            Paragraph("Model Training Pipeline", table_cell),
            Paragraph("<b>extract_landmarks.py:</b> Extracts wrist-centered coordinates from raw images.<br/><b>train_model.py:</b> Trains sequential MLP with Keras & exports quantized .tflite.", table_cell)
        ],
        [
            Paragraph("Root scripts (*.py)", table_cell_bold),
            Paragraph("Desktop Validation", table_cell),
            Paragraph("<b>tracker.py / test_model.py:</b> Desktop OpenCV & webcam test harness for live evaluation before building mobile APKs.", table_cell)
        ],
    ]

    t_repo = Table(repo_data, colWidths=[2.2 * inch, 1.4 * inch, 3.4 * inch])
    t_repo.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
    ]))
    elements.append(t_repo)
    elements.append(Spacer(1, 14))

    # Page Break for clean presentation
    elements.append(PageBreak())

    # -------------------------------------------------------------
    # SECTION 3: END-TO-END DATA PIPELINE
    # -------------------------------------------------------------
    elements.append(Paragraph("3. End-to-End Data Pipeline: Step-by-Step", h1_style))
    elements.append(Paragraph(
        "The live translation pipeline converts photonic data from the camera sensor into syntactic text and speech output "
        "through a sequence of 6 deterministic processing stages:",
        body_style
    ))

    pipeline_stages = [
        ("Phase 1: Sensor Capture (CameraX / VisionCamera)",
         "VisionCamera captures frames via <b>usePhotoOutput</b> configured with balanced quality prioritization. "
         "In AUTO mode, a timer requests a snapshot every 1000ms. In MANUAL mode, the user taps 'Sign Now' to trigger "
         "an on-demand capture. This avoids continuous burst captures while preserving full 60 FPS viewfinder fluidity."),
        ("Phase 2: EXIF Orientation Correction & Adaptive Downsampling",
         "On Android, front-facing camera sensors capture images in landscape orientation (rotated 90 or 270 degrees). "
         "<b>ExpoMediapipeHandsModule.kt</b> queries image dimensions using <b>inJustDecodeBounds</b>, computes an optimal "
         "<b>inSampleSize</b> (scaling down multi-megapixel photos to ~384–512px), and applies an <b>ExifInterface</b> rotation "
         "matrix. This reduces memory allocation by 95% and guarantees upright hand orientation for MediaPipe."),
        ("Phase 3: MediaPipe HandLandmarker Detection",
         "The downsampled ARGB bitmap is passed to <b>com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker</b>. "
         "The detector evaluates palm anchors and regressively predicts the 21 normalized 3D keypoints (x, y, z) in "
         "continuous normalized image space [0.0, 1.0]. The Kotlin module returns a flat array of 63 doubles."),
        ("Phase 4: Wrist-Centered Coordinate Normalization",
         "In <b>handDetector.ts</b>, the 63 coordinates undergo spatial translation. Raw coordinates vary based on hand placement "
         "relative to the camera boundary. The normalizer subtracts Landmark 0 (wrist: x0, y0, z0) from all 21 points:<br/>"
         "&nbsp;&nbsp;&bull; <b>&Delta;x_i = x_i - x_0</b>, &nbsp;&nbsp; <b>&Delta;y_i = y_i - y_0</b>, &nbsp;&nbsp; <b>&Delta;z_i = z_i - z_0</b><br/>"
         "This maps Landmark 0 to origin (0, 0, 0), making the feature vector position-invariant."),
        ("Phase 5: High-Speed TFLite Inference (Fast-TFLite v3)",
         "The normalized Float32Array(63) buffer is fed to <b>activeModel.runSync()</b>. The on-device neural network (a deep "
         "Multi-Layer Perceptron trained in Keras) performs matrix multiplications and ReLU activations, terminating in a "
         "Softmax distribution over all candidate gestures. Inference completes in under 1 millisecond on the device CPU/GPU."),
        ("Phase 6: Visual HUD, Audio Synthesis, and Transcript Tracking",
         "If the top class exceeds the confidence threshold (60%), the UI displays the sign name and score. "
         "<b>HandSkeletonOverlay.tsx</b> draws the 21 joints and 21 connecting bone segments over the live viewfinder. "
         "<b>expo-speech</b> synthesizes the recognized gesture into audible speech, and the word is appended to the transcript "
         "history for sentence composition."),
    ]

    for title, desc in pipeline_stages:
        elements.append(Paragraph(title, h2_style))
        elements.append(Paragraph(desc, body_style))
        elements.append(Spacer(1, 3))

    elements.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # SECTION 4: MAIN FILE CATALOG & ARCHITECTURAL HIGHLIGHTS
    # -------------------------------------------------------------
    elements.append(Paragraph("4. Main File Catalog & Architectural Patterns", h1_style))

    file_catalog = [
        ("LiveTranslationScreen.tsx (Architectural Origin)",
         "<b>File:</b> GesturaMobile/src/screens/LiveTranslationScreen.tsx<br/>"
         "<b>Architectural Pattern: JNI Reference Isolation.</b> VisionCamera hooks (such as useCameraDevice) "
         "maintain active native JNI pointers. In high-frequency computer vision apps, updating React state (e.g., probability bars, "
         "confidence numbers) causes the entire tree to re-render, frequently re-binding camera outputs and leaking native JNI handles. "
         "LiveTranslationScreen solved this by bifurcating the view into: (1) <b>CameraShell</b>, which strictly manages hardware "
         "and never re-renders on prediction updates, and (2) <b>GestureProcessor</b>, which manages high-frequency state."),
        ("index.tsx (Production Viewfinder & Controller)",
         "<b>File:</b> GesturaMobile/src/app/index.tsx<br/>"
         "<b>Role:</b> Unified live translation interface combining multi-model switching (Standard, Indian Sign Language, General), "
         "AUTO/MANUAL operation modes, front/rear camera flipping, flashlight/torch hardware safeguards, and sentence transcript "
         "management (Undo, Speak All, Share/Copy, Clear)."),
        ("HandSkeletonOverlay.tsx (Zero-Dependency Vector Canvas)",
         "<b>File:</b> GesturaMobile/src/components/HandSkeletonOverlay.tsx<br/>"
         "<b>Role:</b> High-performance UI overlay rendering the 21 MediaPipe joints (color-coded by wrist, knuckles, and fingertips) "
         "and 21 connecting bone lines using pure React Native View primitives and rotational matrix transforms. Features an intelligent "
         "target reticle that turns neon green and activates 'HAND LOCKED' upon detection."),
        ("ExpoMediapipeHandsModule.kt (Native Android Processing Engine)",
         "<b>File:</b> GesturaMobile/modules/expo-mediapipe-hands/.../ExpoMediapipeHandsModule.kt<br/>"
         "<b>Role:</b> Native Kotlin module directly integrating Google MediaPipe Tasks Vision. Implements safe asset resolution "
         "(assets/hand_landmarker.task with fallback to internal storage), EXIF portrait orientation rotation, dynamic inSampleSize "
         "downsampling, and explicit bitmap recycling to prevent out-of-memory crashes."),
        ("explore.tsx (Interactive Sign Language Dictionary)",
         "<b>File:</b> GesturaMobile/src/app/explore.tsx<br/>"
         "<b>Role:</b> Educational reference module featuring real-time search, category filtering (Standard, ISL Alphabet A-Z, "
         "Numbers 1-9), and expandable flashcards containing step-by-step finger placement instructions and camera distance tips."),
    ]

    for title, desc in file_catalog:
        elements.append(Paragraph(title, h2_style))
        elements.append(Paragraph(desc, body_style))
        elements.append(Spacer(1, 4))

    # -------------------------------------------------------------
    # SECTION 5: PERFORMANCE OPTIMIZATION SUMMARY
    # -------------------------------------------------------------
    elements.append(Spacer(1, 6))
    elements.append(Paragraph("5. Performance & Reliability Engineering Summary", h1_style))

    perf_text = (
        "<b>Engineering Achievements in Gestura v1.0:</b><br/>"
        "&bull; <b>95% Latency Reduction:</b> Eliminating default 4K UHD still capture in favor of adaptive inSampleSize "
        "downsampling (~384–512px) decreased per-frame processing latency from 800ms down to ~25ms.<br/>"
        "&bull; <b>Hardware Exception Prevention:</b> Camera2 flash exceptions ('no flash unit') on front-facing lenses were "
        "resolved by making torchMode conditional on physical device hardware capabilities (device.hasTorch).<br/>"
        "&bull; <b>Orientation Invariance:</b> Automated EXIF rotation alignment ensures MediaPipe receives upright hands regardless "
        "of front camera sensor orientation.<br/>"
        "&bull; <b>Position Invariance:</b> Wrist-centered vector normalization allows users to sign anywhere inside the viewfinder "
        "frame with consistent high classification confidence."
    )
    perf_table = Table([[Paragraph(perf_text, body_style)]], colWidths=[7.0 * inch])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('LINELEFT', (0, 0), (0, -1), 3.0, colors.HexColor("#0D9488")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    elements.append(perf_table)

    # Build PDF with custom NumberedCanvas
    doc.build(elements, canvasmaker=NumberedCanvas)
    print(f"Successfully generated: {PDF_FILENAME}")

if __name__ == "__main__":
    build_pdf()
