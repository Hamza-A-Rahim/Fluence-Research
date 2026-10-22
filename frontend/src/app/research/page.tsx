import type { Metadata } from "next";
import { ResearchBody } from "@/components/ResearchBody";

export const metadata: Metadata = { title: "Research — Fluence" };

export default function ResearchPage() {
  return <ResearchBody />;
}
