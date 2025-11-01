from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

#FastAPI instance
app = FastAPI()

#Define allowed origins for CORS
origins = [
    "http://localhost:3000",
]

#Add CORS middleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
def read_root():
    """Root endpoint returning a simple JSON response."""
    return {"Hello": "World"}