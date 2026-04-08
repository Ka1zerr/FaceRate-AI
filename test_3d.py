import urllib.request
import json

def post_multipart(url, filename):
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    body = bytearray()
    with open(filename, 'rb') as f: file_content = f.read()
    body.extend(f'--{boundary}\r\n'.encode('utf-8'))
    body.extend(f'Content-Disposition: form-data; name="file"; filename="test.png"\r\n'.encode('utf-8'))
    body.extend(b'Content-Type: image/png\r\n\r\n')
    body.extend(file_content)
    body.extend(f'\r\n--{boundary}--\r\n'.encode('utf-8'))
    req = urllib.request.Request(url, data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    try:
        r = urllib.request.urlopen(req)
        return json.dumps(json.loads(r.read()), indent=2, ensure_ascii=False)
    except Exception as e:
        return str(e)

out1 = post_multipart("http://localhost:8000/analyze", r"C:\Users\Mi\.gemini\antigravity\brain\8de08cf0-d5cb-4ebc-8ab2-168b37459e41\average_face_photo_1773571979996.png")
out2 = post_multipart("http://localhost:8000/analyze", r"C:\Users\Mi\.gemini\antigravity\brain\8de08cf0-d5cb-4ebc-8ab2-168b37459e41\model_test_photo_1773570518130.png")

with open(r"C:\Users\Mi\.gemini\antigravity\scratch\face_app\test_out_3d.txt", "w", encoding="utf-8") as f:
    f.write("=== AVERAGE FACE (3D) ===\n" + out1 + "\n\n=== MODEL FACE (3D) ===\n" + out2)

print("Done! Results saved to test_out_3d.txt")
print("\n=== AVERAGE FACE ===")
print(out1)
print("\n=== MODEL FACE ===")
print(out2)
