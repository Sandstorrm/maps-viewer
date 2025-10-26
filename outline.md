# Map Viewer GUI - Project Outline

## Overview
A GUI application for viewing and navigating CSV-based map files with faction territory visualization.

## Map Format Specification
The CSV files represent maps where each CSV entry is a chunk on the map:
- North is up
- `0` = unexplored area
- `1` = explored and empty area  
- Other values = name of the faction that controls that chunk

## Core Features

### 1. Tab-Based Map Interface
- **Tab List**: Display tabs for each map file in `/maps` directory
- **Tab Names**: Use CSV filename (without extension) as world name
- **Dynamic Loading**: Load available maps from `/maps` directory on startup
- **Tab Switching**: Allow switching between different world maps

### 2. Map Navigation
- **Pan**: Click and drag to pan around the map
- **Zoom**: Mouse wheel or zoom controls to zoom in/out of the map
- **Hover Information**: Display chunk coordinates and block position on hover
  - Show chunk coordinates (x, y)
  - Calculate block position: chunk coordinates × 16
  - Display faction name if chunk is claimed

### 3. Map Visualization
- **Chunk Rendering**: Render each CSV entry as a map
- **Color Coding**: 
  - Unexplored areas (0): Dark/black
  - Empty explored areas (1): Light gray/white
  - Faction territories: Unique colors per faction

### 4. Search Functionality
- **Faction Search**: Input field to search for specific faction names
- **Highlight Results**: Highlight all chunks belonging to searched faction
- **Tab Filtering**: Limit tab selection to maps containing the searched faction
- **Clear Search**: Button to clear search and show all maps

### 5. File Management
- **Refresh Button**: Reload all CSV files from `/maps` directory
- **Live Updates**: Allow updating maps while program is running (refresh button)
