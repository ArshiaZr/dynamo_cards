"use client";

import FlashCard from "@/components/FalshCard";
import styles from "@/styles/page.module.scss";
import axios from "axios";
import { useState } from "react";
import { LuLoader2 } from "react-icons/lu";

export default function Home() {
  const [url, setUrl] = useState("");
  const [keyConcepts, setKeyConcepts] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleURLChange = (e) => {
    setUrl(e.target.value);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    retrive_analysis();
  };

  const retrive_analysis = async () => {
    if (!url) {
      console.error("URL is empty");
      return;
    }
    setLoading(true);
    try {
      const response = await axios.post(
        process.env.NEXT_PUBLIC_SERVER_URL + "analyze_video",
        { youtube_link: url }
      );
      if (response.data.error) {
        console.error(response.data.error);
        setKeyConcepts([]);
        return;
      }
      if (
        !response.data.key_concepts ||
        response.data.key_concepts.length === 0
      ) {
        console.error("No key concepts found");
        return;
      }
      console.log(response.data.key_concepts);
      setKeyConcepts(response.data.key_concepts);
    } catch {
      console.error("Failed to retrieve analysis");
      setKeyConcepts([]);
    }
    setLoading(false);
  };

  const discardFlashCard = (index) => {
    const newKeyConcepts = keyConcepts.filter((_, i) => i !== index);
    setKeyConcepts(newKeyConcepts);
  };

  return (
    <main id={styles.home}>
      <div className={styles.wrapper}>
        <div className={styles.inputWrapper}>
          <div className={styles.title}>
            Enter the Youtube URL to get Flash Cards
          </div>
          <form onSubmit={handleSubmit}>
            <input
              type="text"
              placeholder="https://youtube.com/"
              onChange={handleURLChange}
            />
            <button type="submit">Submit</button>
          </form>
        </div>

        {loading && (
          <div className={styles.loading}>
            <LuLoader2 className={styles.spinner} />
          </div>
        )}

        <div className={styles.flashCardsContainer}>
          {keyConcepts.map((concept, index) => (
            <div key={index}>
              <FlashCard
                concept={concept.concept}
                description={concept.description}
                discard={() => discardFlashCard(index)}
              />
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
