from chat_engine import responder_pregunta

print("🤖 Asistente Analítico Energético")
print("Escribe tu pregunta (salir para terminar)\n")

while True:
    pregunta = input("Usuario: ")
    if pregunta.lower() == "salir":
        break

    respuesta = responder_pregunta(pregunta)
    print("\nAsistente:", respuesta, "\n")
