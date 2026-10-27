import { redirect } from "next/navigation";
import { hasSession } from "@/lib/api";

export default async function Home() {
  redirect((await hasSession()) ? "/feed" : "/login");
}
