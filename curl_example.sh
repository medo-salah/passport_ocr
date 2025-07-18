curl -X POST "http://localhost:8000/process" \
  -H "accept: application/json" \
  -F "file=@/path/to/your/passport.jpg" \
  -F 'options={"language":"eng","enable_face_detection":true,"enable_document_classification":true}'