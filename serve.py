import http.server
import socketserver
import webbrowser
import sys
import os

PORT = 8000
FALLBACK_PORT = 8080

def start_server(port):
    Handler = http.server.SimpleHTTPRequestHandler
    
    # Assicuriamoci che il server serva i file dalla cartella corrente
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            print(f"Server DealHunter avviato su http://localhost:{port}")
            print("Per fermare il server: CTRL+C")
            
            # Auto-apertura browser
            webbrowser.open(f"http://localhost:{port}/index.html")
            
            httpd.serve_forever()
    except OSError as e:
        if e.errno == 98 or e.errno == 10048: # Indirizzo già in uso
            if port == PORT:
                print(f"Porta {PORT} occupata, provo porta {FALLBACK_PORT}...")
                start_server(FALLBACK_PORT)
            else:
                print(f"Errore: Anche la porta {FALLBACK_PORT} è occupata. Kill Switch attivato.")
                sys.exit(1)
        else:
            print(f"Errore critico durante l'avvio del server: {e}")
            sys.exit(1)

if __name__ == "__main__":
    start_server(PORT)
