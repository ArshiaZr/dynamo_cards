"use client";

import styles from "@/styles/page.module.scss";
import axios from "axios";
import { useState } from "react";

export default function Home() {
  const [url, setUrl] = useState("");
  const [response, setResponse] = useState("");

  const handleURLChange = (e) => {
    setUrl(e.target.value);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    retrive_analysis();
  };
  const retrive_analysis = async () => {
    try {
      const response = await axios.post(
        process.env.NEXT_PUBLIC_SERVER_URL + "analyze_video",
        { youtube_link: url }
      );
      setResponse(JSON.stringify(response.data));
    } catch {}
  };

  return (
    <main id={styles.home}>
      <div className={styles.wrapper}>
        <h3>Enter the Youtube URL</h3>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            placeholder="https://youtube.com/"
            onChange={handleURLChange}
          />
          <button type="submit">Submit</button>
        </form>

        <div>Response: {response}</div>
      </div>
    </main>
  );
}
