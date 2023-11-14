from fastapi.responses import JSONResponse

def JSONResponseHandler(data):
    try:
        return JSONResponse(data)
    except TypeError as e:
        print(e)
        print(data)
        return {}