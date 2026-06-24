import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Nav } from "@/components/Nav";
import Landing from "@/pages/Landing";
import Demo1 from "@/pages/Demo1";
import Demo2 from "@/pages/Demo2";

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-background text-foreground">
        <Nav />
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/demo1" element={<Demo1 />} />
          <Route path="/demo2" element={<Demo2 />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
