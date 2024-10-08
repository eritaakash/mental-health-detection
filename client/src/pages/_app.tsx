import { AppProps } from 'next/app';
import '../styles/global.css';

const MyApp: React.FC<AppProps> = ({ Component, pageProps }) =>
    <Component {...pageProps} />;

export default MyApp;