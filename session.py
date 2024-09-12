from uuid import UUID, uuid4

from pydantic import BaseModel
from fastapi import HTTPException, APIRouter, Request, Response, Depends
from fastapi_sessions.backends.implementations import InMemoryBackend
from fastapi_sessions.session_verifier import SessionVerifier
from fastapi_sessions.frontends.implementations import SessionCookie, CookieParameters


class SessionData(BaseModel):
    username: str


class BasicVerifier(SessionVerifier[UUID, SessionData]):
    def __init__(
        self,
        *,
        identifier: str,
        auto_error: bool,
        backend: InMemoryBackend[UUID, SessionData],
        auth_http_exception: HTTPException,
    ):
        self._identifier = identifier
        self._auto_error = auto_error
        self._backend = backend
        self._auth_http_exception = auth_http_exception

    @property
    def identifier(self):
        return self._identifier

    @property
    def backend(self):
        return self._backend

    @property
    def auto_error(self):
        return self._auto_error

    @property
    def auth_http_exception(self):
        return self._auth_http_exception

    def verify_session(self, model: SessionData) -> bool:
        """If the session exists, it is valid"""
        return True


class SessionManager():
    def __init__(self):
        self._cookie_params = CookieParameters()
        self.session_user_data = {}

        # Uses UUID
        self._cookie = SessionCookie(
            cookie_name="cookie",
            identifier="general_verifier",
            auto_error=True,
            secret_key="DONOTUSE",
            cookie_params=self._cookie_params,
        )
        self._backend = InMemoryBackend[UUID, SessionData]()
        self._verifier = BasicVerifier(
            identifier="general_verifier",
            auto_error=True,
            backend=self._backend,
            auth_http_exception=HTTPException(status_code=403, detail="invalid session"),
        )

    @property
    def cookie_params(self):
        return self._cookie_params

    @property
    def cookie(self):
        return self._cookie

    @property
    def backend(self):
        return self._backend

    @property
    def verifier(self):
        return self._verifier

    def save_data(self, session_id, data):
        self.session_user_data[session_id] = data

    def get_data(self, session_id):
        try:
            return self.session_user_data[session_id]
        except KeyError:
            return None

    def delete_data(self, session_id):
        try:
            del self.session_user_data[session_id]
        except KeyError:
            pass


session_manager = SessionManager()

session_router = APIRouter(prefix="/api/session", tags=["Session"])

@session_router.post("/create/{name}")
async def create_session(name: str, response: Response):
    session = uuid4()
    data = SessionData(username=name)
    await session_manager.backend.create(session, data)
    session_manager.cookie.attach_to_response(response, session)
    return {
        "success": True,
        "username": name,
        "session": str(session)
    }    # do not use JSONResponse, cookie cannot be attached to JSONResponse

@session_router.get("/current", dependencies=[Depends(session_manager.cookie)])
async def get_session(session_data: SessionData = Depends(session_manager.verifier)):
    return {
        "success": True,
        "username": session_data.username
    }

@session_router.delete("/current")
async def delete_session(response: Response, session_id: UUID = Depends(session_manager.cookie)):
    await session_manager.backend.delete(session_id)
    session_manager.cookie.delete_from_response(response)
    session_manager.delete_data(session_id)
    return {
        "success": True
    }

@session_router.post("/data")
async def save_session_data(request: Request, session_id: UUID = Depends(session_manager.cookie)):
    data = await request.json()
    session_manager.save_data(session_id, data)
    return {
        "success": True
    }

@session_router.get("/data")
async def get_session_data(session_id: UUID = Depends(session_manager.cookie)):
    data = session_manager.get_data(session_id)
    return {
        "success": True,
        "data": data
    }

@session_router.delete("/data")
async def delete_session_data(session_id: UUID = Depends(session_manager.cookie)):
    session_manager.delete_data(session_id)
    return {
        "success": True
    }
