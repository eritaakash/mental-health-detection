import React, { useState, useEffect } from 'react';
import Icons from 'feather-icons-react';

const HomePage: React.FC = () => {

    const [status, setStatus] = useState('None');
    const [text, setText] = useState('');

    const [pending, setPending] = useState(false);

    const classifySentiment = async () => {
        setPending(true);

        if (text.trim() === '') {
            alert('Please enter some text');

            setPending(false);
            return;
        }

        const url = 'https://mental-sentiment.onrender.com/predict';

        try {
            const res = await fetch(url, {
                method: 'POST',
                headers: {
                    'Accept': '*/*',
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    data: [text],
                }),
            });

            if (res.ok) {
                const data = await res.json();
                setStatus(data.predictions[0]);

                setPending(false);
            } else {
                setStatus('Error: Unable to classify');
                setPending(false);
            }
        } catch (error) {
            console.error('Error:', error);
            setStatus('Error: Something went wrong');

            setPending(false);
        }
    };

    return (
        <main>
            <h1>Mental Health Classification</h1>
            <p>Using Natural Language Processing and Machine Learning, from your texts!</p>

            <section>
                <div>
                    <textarea
                        placeholder={'Write your text here.'}
                        onChange={(e) => setText(e.target.value)}></textarea>

                        {
                            pending 
                            ? <span><Icons icon='loader' />Checking...</span>
                            : <span>Status: <strong>{status}</strong></span>
                        }
                </div>

                {
                    pending 
                    ? <button><Icons icon='loader' />Checking...</button>
                    : <button onClick={classifySentiment}>Classify!</button>
                }
            </section>
        </main>
    )
}

export default HomePage;