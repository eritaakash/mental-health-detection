import React, { useState, useEffect } from 'react';
import Icons from 'feather-icons-react';

const HomePage: React.FC = () => {

    const [status, setStatus] = useState('None');
    const [text, setText] = useState('');

    const classifySentiment = async () => {
        const url = 'https://mental-sentiment.onrender.com/predict';

        const res = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                data: [text]
            })
        });

        const data = await res.json();
        setStatus(data.predictions[0]);
    }
    
    return (
        <main>
            <h1>Mental Health Classification</h1>
            <p>Using Natural Language Processing and Machine Learning, from your texts!</p>

            <section>
                <div>
                    <textarea placeholder={'Write your text here.'}></textarea>
                    <span>Status: <strong>{status}</strong></span>
                </div>

                <button onClick={classifySentiment}>Classify!</button>
            </section>
        </main>
    )
}

export default HomePage;