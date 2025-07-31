#!/usr/bin/env python3

import uvicorn
from sejm_whiz.web_api.core import get_app

app = get_app()


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
