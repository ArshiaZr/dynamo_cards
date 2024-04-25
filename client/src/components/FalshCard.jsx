import styles from "@/styles/flashcard.module.scss";

export default function FlashCard({ concept, description, discard }) {
  return (
    <div className={styles.flashcard}>
      <div className={styles.contents}>
        <div className={styles.concept}>{concept}</div>
        <div className={styles.description}>{description}</div>
      </div>
      <button className={styles.discard} onClick={discard}>
        Discard
      </button>
    </div>
  );
}
