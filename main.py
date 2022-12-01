# uvicorn main:app --reload
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import covid19_v1, seird

#CORS
origins = [
    "http://localhost:3000",
    "*"
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(covid19_v1.router)
app.include_router(seird.router)

# @app.get("/")
# def read_root():
#     return {"data": "Hello World"}

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}

if __name__ == "__main__":
    # app.run(debug=False)
    app.run(host="0.0.0.0",port=8080)
