# curl -X POST -H "Content-Type: application/json" -d '{"text":"Hello, World!"}' http://localhost:8080
import http.server
import socketserver
import json
from urllib.parse import parse_qs

class SimpleRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        parsed_data = json.loads(post_data)

        if 'text' in parsed_data:
            input_text = parsed_data['text']
            reversed_text = input_text[::-1]
            response = {'reversed_text': reversed_text}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_error(400, 'Bad Request: JSON object with "text" key expected')

def run(server_class=http.server.HTTPServer, handler_class=SimpleRequestHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting server on port {port}')
    httpd.serve_forever()

if __name__ == '__main__':
    run()
