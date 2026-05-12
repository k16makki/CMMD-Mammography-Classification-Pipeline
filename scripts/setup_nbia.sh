#!/bin/bash

set -e

echo " Checking TCIA Data Retriever setup..."

APPIMAGE="tools/NBIADataRetriever/TCIA_Data_Retriever-x86_64.AppImage"

if [ ! -f "$APPIMAGE" ]; then
    echo " AppImage not found!"
    echo " Download it from TCIA website and place it here:"
    echo "$APPIMAGE"
    exit 1
fi

chmod +x "$APPIMAGE"

echo "✅ TCIA Data Retriever ready"
