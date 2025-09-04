#!/bin/bash

make evaluations
uvicorn app.api:app --host "0.0.0.0"
