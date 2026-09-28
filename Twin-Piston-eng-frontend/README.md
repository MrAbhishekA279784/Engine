# Twin-Piston Aero Engine Digital Twin (Frontend)

An interactive, aerospace-grade 3D Digital Twin and telemetry intelligence system designed for real-time health monitoring, predictive maintenance, and mission reliability enhancement of twin-piston aero engines used in Medium-Altitude Long-Endurance (MALE) UAVs.

---

## ✈️ System Overview

The **Twin-Piston Digital Twin** provides engineers, fleet operators, and mission commanders with deep situational awareness and prognostics for twin-piston aircraft powerplants. The application unifies high-fidelity 3D spatial visualization with real-time telemetry streaming, subsystem decomposition, and explainable AI insights.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        AEROTWIN SIGHT SYSTEM                           │
├──────────────────┬─────────────────────────────┬───────────────────────┤
│  LEFT PANEL      │       CENTER VIEWPORT       │      RIGHT PANEL      │
│  Navigation      │  Interactive 3D Engine      │  Component Telemetry  │
│  System Status   │  Exploded View Kinematics   │  Health Gauges (CHT)  │
│  Mission Context │  Subsystem Highlighting     │  Degradation Vectors  │
├──────────────────┴─────────────────────────────┴───────────────────────┤
│                             BOTTOM DOCK                                │
│       Real-Time Sensor Telemetry Trends • Vibration • Oil Flow • RPM   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🌟 Key Features

### 1. Interactive 3D Digital Twin
- **Dynamic 3D Kinematics**: Complete 360° orbit, smooth camera transitions, zoom, and spatial pan.
- **Exploded View Architecture**: Smooth kinematic disassembly/reassembly of critical components (Cylinders, Pistons, Connecting Rods, Crankshaft, Cylinder Heads, Intake/Exhaust Systems, Valves, and FADEC).
- **Spatial Subsystem Focus**: Click-to-focus on individual engine parts with synchronized health state transitions.

### 2. Real-Time Telemetry & Health Monitoring
- Continuous sensor ingestion and visualization for:
  - **RPM & Power Delivery**
  - **Cylinder Head Temperature (CHT)** & **Exhaust Gas Temperature (EGT)**
  - **Oil Pressure & Temperature**
  - **Multi-Axis Vibration Metrics**
  - **Fuel Flow & Manifold Absolute Pressure (MAP)**

### 3. Predictive Diagnostics & Health Metrics
- **Remaining Useful Life (RUL)** estimation.
- **Anomaly Detection & Severity Index** tracking.
- **Fault Probability Decomposition** per component.
- **Mission Reliability Forecasting** with Safe / Caution / Unsafe risk classifications.

---

## 🛠️ Technology Stack

- **Framework**: [React 19](https://react.dev/) + [TypeScript](https://www.typescriptlang.org/)
- **Routing & State**: [TanStack Router](https://tanstack.com/router) & [TanStack Start](https://tanstack.com/start)
- **3D Graphics & WebGL**: [Three.js](https://threejs.org/) + [@react-three/fiber](https://r3f.docs.pmnd.rs/) + [@react-three/drei](https://github.com/pmndrs/drei)
- **Styling**: [Tailwind CSS v4](https://tailwindcss.com/)
- **Data Visualization**: [Recharts](https://recharts.org/)
- **Icons & UI Primitives**: [Lucide React](https://lucide.dev/), Radix UI

---

## 🚀 Getting Started

### Prerequisites
- **Node.js**: v18.0 or higher
- **Package Manager**: `npm`, `pnpm`, or `bun`

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Rehan-roid/Twin-Piston-eng-frontend.git
   cd Twin-Piston-eng-frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```

4. **Access the application**:
   Open [http://localhost:8080](http://localhost:8080) (or the port displayed in your terminal) in your browser.

---

## 📦 Build & Production

To compile the production build:

```bash
npm run build
```

To preview the production build locally:

```bash
npm run preview
```

---

## 📂 Project Structure

```
├── public/
│   └── twin-piston-engine.glb     # Optimized 3D digital twin model
├── src/
│   ├── components/
│   │   ├── digital-twin/          # 3D Viewer, Gauges, Exploded View, Telemetry
│   │   └── ui/                    # Reusable UI primitives
│   ├── data/                      # Mock engine telemetry & baseline parameters
│   ├── lib/                       # Utility functions, kinematics, error handling
│   ├── routes/                    # TanStack file-based application routes
│   └── types/                     # TypeScript type definitions for engine domain
├── tools/
│   └── mesh-analysis/             # Mesh structure validation & hierarchy analysis tools
└── vite.config.ts                 # Build and plugin configuration
```

---

## 📄 License

This project is licensed under the MIT License.
