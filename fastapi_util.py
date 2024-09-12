from fastapi.responses import JSONResponse


def JSONResponseHandler(data):  # pylint: disable=invalid-name
    try:
        return JSONResponse(data)
    except TypeError as e:
        print(e)
        print(data)
        return {}
